#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:10:44 2017
@author: wroscoe
remotes.py
The client and web server needed to control a car remotely.
"""


import os
import json
import logging
import time
import asyncio

import requests
from tornado.ioloop import IOLoop
from tornado.web import Application, RedirectHandler, StaticFileHandler, \
    RequestHandler
from tornado.httpserver import HTTPServer
import tornado.gen
import tornado.websocket
from socket import gethostname

from ... import utils

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, \
        RTCConfiguration, RTCIceServer
    from av import VideoFrame
    WEBRTC_AVAILABLE = True
except ImportError:
    RTCPeerConnection = None
    RTCSessionDescription = None
    VideoStreamTrack = object
    RTCConfiguration = None
    RTCIceServer = None
    VideoFrame = None
    WEBRTC_AVAILABLE = False

logger = logging.getLogger(__name__)


class DonkeyVideoStreamTrack(VideoStreamTrack):
    """Stream the latest frame from LocalWebController to a WebRTC peer."""

    def __init__(self, app):
        super().__init__()
        self._app = app

    async def recv(self):
        if not WEBRTC_AVAILABLE:
            raise RuntimeError("WebRTC dependencies are not installed")

        pts, time_base = await self.next_timestamp()
        frame_arr = getattr(self._app, 'img_arr', None)
        if frame_arr is None:
            await asyncio.sleep(0.03)
            frame_arr = utils.load_image_sized(
                os.path.join(self._app.static_file_path, "img_placeholder.jpg"),
                160,
                120,
                3,
            )

        frame = VideoFrame.from_ndarray(frame_arr, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


class RemoteWebServer():
    '''
    A controller that repeatedly polls a remote webserver and expects
    the response to be angle, throttle and drive mode.
    '''

    def __init__(self, remote_url, connection_timeout=.25):

        self.control_url = remote_url
        self.time = 0.
        self.angle = 0.
        self.throttle = 0.
        self.mode = 'user'
        self.mode_latch = None
        self.recording = False
        # use one session for all requests
        self.session = requests.Session()

    def update(self):
        '''
        Loop to run in separate thread the updates angle, throttle and
        drive mode.
        '''

        while True:
            # get latest value from server
            self.angle, self.throttle, self.mode, self.recording = self.run()

    def run_threaded(self):
        '''
        Return the last state given from the remote server.
        '''
        return self.angle, self.throttle, self.mode, self.recording

    def run(self):
        '''
        Posts current car sensor data to webserver and returns
        angle and throttle recommendations.
        '''

        data = {}
        response = None
        while response is None:
            try:
                response = self.session.post(self.control_url,
                                             files={'json': json.dumps(data)},
                                             timeout=0.25)

            except requests.exceptions.ReadTimeout as err:
                print("\n Request took too long. Retrying")
                # Lower throttle to prevent runaways.
                return self.angle, self.throttle * .8, None

            except requests.ConnectionError as err:
                # try to reconnect every 3 seconds
                print("\n Vehicle could not connect to server. Make sure you've " +
                    "started your server and you're referencing the right port.")
                time.sleep(3)

        data = json.loads(response.text)
        angle = float(data['angle'])
        throttle = float(data['throttle'])
        drive_mode = str(data['drive_mode'])
        recording = bool(data['recording'])

        return angle, throttle, drive_mode, recording

    def shutdown(self):
        pass


class LocalWebController(tornado.web.Application):

    def __init__(self, port=8887, mode='user', webrtc_enabled=True,
                 webrtc_ice_servers=None):
        """
        Create and publish variables needed on many of
        the web handlers.
        """
        logger.info('Starting Donkey Server...')

        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = mode
        self.mode_latch = None
        self.recording = False
        self.recording_latch = None
        self.buttons = {}  # latched button values for processing
        self.webrtc_enabled = bool(webrtc_enabled)
        self.webrtc_ice_servers = list(webrtc_ice_servers or [])

        self.port = port

        self.num_records = 0
        self.wsclients = []
        self.webrtc_peers = set()
        self.loop = None

        # Optional FIRA telemetry values shown in the web UI.
        self.fira_obstacle_severity = 0.0
        self.fira_lane_confidence = 0.0
        self.fira_safety_failsafe_active = False
        self.fira_safety_lane_weight = 0.0
        self.fira_safety_obstacle_weight = 0.0
        self.fira_safety_curve_factor = 1.0
        self.fira_safety_speed_limit_factor = 1.0
        self.fira_drive_state = "IDLE"
        self.fira_drive_state_throttle_cap = 1.0
        self.fira_competition_right_lane_score = 0.0
        self.fira_competition_checkpoint_count = 0
        self.fira_competition_checkpoint_progress = 0.0
        self.fira_competition_lane_violations = 0
        self.fira_competition_active_frames = 0
        self.fira_competition_compliance_score = 0.0
        self.fira_competition_compliance_ready = False


        handlers = [
            (r"/", RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveAPI),
            (r"/wsDrive", WebSocketDriveAPI),
            (r"/wsCalibrate", WebSocketCalibrateAPI),
            (r"/calibrate", CalibrateHandler),
            (r"/video", VideoAPI),
            (r"/webrtc/config", WebRTCConfigAPI),
            (r"/webrtc/health", WebRTCHealthAPI),
            (r"/webrtc/offer", WebRTCOfferAPI),
            (r"/wsTest", WsTest),

            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path}),
        ]

        settings = {'debug': True}
        super().__init__(handlers, **settings)
        logger.info(f"You can now go to {gethostname()}.local:{port} to "
                    f"drive your car.")

    def update(self):
        """ Start the tornado webserver. """
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.listen(self.port)
        self.loop = IOLoop.instance()
        self.loop.start()

    def update_wsclients(self, data):
        if data:
            for wsclient in self.wsclients:
                try:
                    data_str = json.dumps(data)
                    logger.debug(f"Updating web client: {data_str}")
                    wsclient.write_message(data_str)
                except Exception as e:
                    logger.warning("Error writing websocket message",
                                   exc_info=e)
                    pass

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def run_threaded(self,
                     img_arr=None,
                     num_records=0,
                     mode=None,
                     recording=None,
                     fira_obstacle_severity=None,
                     fira_lane_confidence=None,
                     fira_safety_failsafe_active=None,
                     fira_safety_lane_weight=None,
                     fira_safety_obstacle_weight=None,
                     fira_safety_curve_factor=None,
                     fira_safety_speed_limit_factor=None,
                     fira_drive_state=None,
                     fira_drive_state_throttle_cap=None,
                     fira_competition_right_lane_score=None,
                     fira_competition_checkpoint_count=None,
                     fira_competition_checkpoint_progress=None,
                     fira_competition_lane_violations=None,
                     fira_competition_active_frames=None,
                     fira_competition_compliance_score=None,
                     fira_competition_compliance_ready=None):
        """
        :param img_arr: current camera image or None
        :param num_records: current number of data records
        :param mode: default user/mode
        :param recording: default recording mode
        """
        self.img_arr = img_arr
        self.num_records = num_records

        #
        # enforce defaults if they are not none.
        #
        changes = {}
        if mode is not None and self.mode != mode:
            self.mode = mode
            changes["driveMode"] = self.mode
        if self.mode_latch is not None:
            self.mode = self.mode_latch
            self.mode_latch = None
            changes["driveMode"] = self.mode
        if recording is not None and self.recording != recording:
            self.recording = recording
            changes["recording"] = self.recording
        if self.recording_latch is not None:
            self.recording = self.recording_latch;
            self.recording_latch = None;
            changes["recording"] = self.recording;

        if fira_obstacle_severity is not None:
            value = self._as_float(fira_obstacle_severity, 0.0)
            if self.fira_obstacle_severity != value:
                self.fira_obstacle_severity = value
                changes["firaObstacleSeverity"] = value
        if fira_lane_confidence is not None:
            value = self._as_float(fira_lane_confidence, 0.0)
            if self.fira_lane_confidence != value:
                self.fira_lane_confidence = value
                changes["firaLaneConfidence"] = value
        if fira_safety_failsafe_active is not None:
            value = bool(fira_safety_failsafe_active)
            if self.fira_safety_failsafe_active != value:
                self.fira_safety_failsafe_active = value
                changes["firaFailsafeActive"] = value
        if fira_safety_lane_weight is not None:
            value = self._as_float(fira_safety_lane_weight, 0.0)
            if self.fira_safety_lane_weight != value:
                self.fira_safety_lane_weight = value
                changes["firaLaneWeight"] = value
        if fira_safety_obstacle_weight is not None:
            value = self._as_float(fira_safety_obstacle_weight, 0.0)
            if self.fira_safety_obstacle_weight != value:
                self.fira_safety_obstacle_weight = value
                changes["firaObstacleWeight"] = value
        if fira_safety_curve_factor is not None:
            value = self._as_float(fira_safety_curve_factor, 1.0)
            if self.fira_safety_curve_factor != value:
                self.fira_safety_curve_factor = value
                changes["firaCurveFactor"] = value
        if fira_safety_speed_limit_factor is not None:
            value = self._as_float(fira_safety_speed_limit_factor, 1.0)
            if self.fira_safety_speed_limit_factor != value:
                self.fira_safety_speed_limit_factor = value
                changes["firaSpeedLimitFactor"] = value
        if fira_drive_state is not None:
            value = str(fira_drive_state)
            if self.fira_drive_state != value:
                self.fira_drive_state = value
                changes["firaDriveState"] = value
        if fira_drive_state_throttle_cap is not None:
            value = self._as_float(fira_drive_state_throttle_cap, 1.0)
            if self.fira_drive_state_throttle_cap != value:
                self.fira_drive_state_throttle_cap = value
                changes["firaDriveStateThrottleCap"] = value
        if fira_competition_right_lane_score is not None:
            value = self._as_float(fira_competition_right_lane_score, 0.0)
            if self.fira_competition_right_lane_score != value:
                self.fira_competition_right_lane_score = value
                changes["firaCompetitionRightLaneScore"] = value
        if fira_competition_checkpoint_count is not None:
            value = int(self._as_float(fira_competition_checkpoint_count, 0.0))
            if self.fira_competition_checkpoint_count != value:
                self.fira_competition_checkpoint_count = value
                changes["firaCompetitionCheckpointCount"] = value
        if fira_competition_checkpoint_progress is not None:
            value = self._as_float(fira_competition_checkpoint_progress, 0.0)
            if self.fira_competition_checkpoint_progress != value:
                self.fira_competition_checkpoint_progress = value
                changes["firaCompetitionCheckpointProgress"] = value
        if fira_competition_lane_violations is not None:
            value = int(self._as_float(fira_competition_lane_violations, 0.0))
            if self.fira_competition_lane_violations != value:
                self.fira_competition_lane_violations = value
                changes["firaCompetitionLaneViolations"] = value
        if fira_competition_active_frames is not None:
            value = int(self._as_float(fira_competition_active_frames, 0.0))
            if self.fira_competition_active_frames != value:
                self.fira_competition_active_frames = value
                changes["firaCompetitionActiveFrames"] = value
        if fira_competition_compliance_score is not None:
            value = self._as_float(fira_competition_compliance_score, 0.0)
            if self.fira_competition_compliance_score != value:
                self.fira_competition_compliance_score = value
                changes["firaCompetitionComplianceScore"] = value
        if fira_competition_compliance_ready is not None:
            value = bool(fira_competition_compliance_ready)
            if self.fira_competition_compliance_ready != value:
                self.fira_competition_compliance_ready = value
                changes["firaCompetitionComplianceReady"] = value

        # Send record count to websocket clients
        if (self.num_records is not None and self.recording is True):
            if self.num_records % 10 == 0:
                changes['num_records'] = self.num_records

        #
        # get latched button presses then clear button presses
        # Next iteration will clear press in memory
        #
        buttons = self.buttons
        self.buttons = {}
        for button, pressed in buttons.items():
            if pressed:
                self.buttons[button] = False

        # if there were changes, then send to web client
        if changes and self.loop is not None:
            logger.debug(str(changes))
            self.loop.add_callback(lambda: self.update_wsclients(changes))

        return self.angle, self.throttle, self.mode, self.recording, buttons

    def run(self,
            img_arr=None,
            num_records=0,
            mode=None,
            recording=None,
            fira_obstacle_severity=None,
            fira_lane_confidence=None,
            fira_safety_failsafe_active=None,
            fira_safety_lane_weight=None,
            fira_safety_obstacle_weight=None,
            fira_safety_curve_factor=None,
            fira_safety_speed_limit_factor=None,
            fira_drive_state=None,
            fira_drive_state_throttle_cap=None,
            fira_competition_right_lane_score=None,
            fira_competition_checkpoint_count=None,
            fira_competition_checkpoint_progress=None,
            fira_competition_lane_violations=None,
            fira_competition_active_frames=None,
            fira_competition_compliance_score=None,
            fira_competition_compliance_ready=None):
        return self.run_threaded(
            img_arr,
            num_records,
            mode,
            recording,
            fira_obstacle_severity,
            fira_lane_confidence,
            fira_safety_failsafe_active,
            fira_safety_lane_weight,
            fira_safety_obstacle_weight,
            fira_safety_curve_factor,
            fira_safety_speed_limit_factor,
            fira_drive_state,
            fira_drive_state_throttle_cap,
            fira_competition_right_lane_score,
            fira_competition_checkpoint_count,
            fira_competition_checkpoint_progress,
            fira_competition_lane_violations,
            fira_competition_active_frames,
            fira_competition_compliance_score,
            fira_competition_compliance_ready,
        )

    def shutdown(self):
        pass


class DriveAPI(RequestHandler):

    def get(self):
        data = {}
        self.render("templates/vehicle.html", **data)

    def post(self):
        '''
        Receive post requests as user changes the angle
        and throttle of the vehicle on a the index webpage
        '''
        data = tornado.escape.json_decode(self.request.body)

        if data.get('angle') is not None:
            self.application.angle = data['angle']
        if data.get('throttle') is not None:
            self.application.throttle = data['throttle']
        if data.get('drive_mode') is not None:
            self.application.mode = data['drive_mode']
        if data.get('recording') is not None:
            self.application.recording = data['recording']
        if data.get('buttons') is not None:
            latch_buttons(self.application.buttons, data['buttons'])


class WsTest(RequestHandler):
    def get(self):
        data = {}
        self.render("templates/wsTest.html", **data)


class CalibrateHandler(RequestHandler):
    """ Serves the calibration web page"""
    async def get(self):
        await self.render("templates/calibrate.html")


def latch_buttons(buttons, pushes):
    """
    Latch button pushes
    buttons: the latched values
    pushes: the update value
    """
    if pushes is not None:
        #
        # we got button pushes.
        # - we latch the pushed buttons so we can process the push
        # - after it is processed we clear it
        #
        for button in pushes:
            # if pushed, then latch it
            if pushes[button]:
                buttons[button] = True


class WebSocketDriveAPI(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        logger.info("New client connected")
        self.application.wsclients.append(self)

    def on_message(self, message):
        data = json.loads(message)
        self.application.angle = data.get('angle', self.application.angle)
        self.application.throttle = data.get('throttle', self.application.throttle)
        if data.get('drive_mode') is not None:
            self.application.mode = data['drive_mode']
            self.application.mode_latch = self.application.mode
        if data.get('recording') is not None:
            self.application.recording = data['recording']
            self.application.recording_latch = self.application.recording
        if data.get('buttons') is not None:
            latch_buttons(self.application.buttons, data['buttons'])

    def on_close(self):
        logger.info("Client disconnected")
        self.application.wsclients.remove(self)


class WebSocketCalibrateAPI(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        logger.info("New client connected")

    def on_message(self, message):
        logger.info(f"wsCalibrate {message}")
        data = json.loads(message)
        if 'throttle' in data:
            print(data['throttle'])
            self.application.throttle = data['throttle']

        if 'angle' in data:
            print(data['angle'])
            self.application.angle = data['angle']

        if 'config' in data:
            config = data['config']
            if self.application.drive_train_type == "PWM_STEERING_THROTTLE" \
                or self.application.drive_train_type == "I2C_SERVO":
                if 'STEERING_LEFT_PWM' in config:
                    self.application.drive_train['steering'].left_pulse = config['STEERING_LEFT_PWM']

                if 'STEERING_RIGHT_PWM' in config:
                    self.application.drive_train['steering'].right_pulse = config['STEERING_RIGHT_PWM']

                if 'THROTTLE_FORWARD_PWM' in config:
                    self.application.drive_train['throttle'].max_pulse = config['THROTTLE_FORWARD_PWM']

                if 'THROTTLE_STOPPED_PWM' in config:
                    self.application.drive_train['throttle'].zero_pulse = config['THROTTLE_STOPPED_PWM']

                if 'THROTTLE_REVERSE_PWM' in config:
                    self.application.drive_train['throttle'].min_pulse = config['THROTTLE_REVERSE_PWM']

            elif self.application.drive_train_type == "MM1":
                if ('MM1_STEERING_MID' in config) and (config['MM1_STEERING_MID'] != 0):
                        self.application.drive_train.STEERING_MID = config['MM1_STEERING_MID']
                if ('MM1_MAX_FORWARD' in config) and (config['MM1_MAX_FORWARD'] != 0):
                        self.application.drive_train.MAX_FORWARD = config['MM1_MAX_FORWARD']
                if ('MM1_MAX_REVERSE' in config) and (config['MM1_MAX_REVERSE'] != 0):
                    self.application.drive_train.MAX_REVERSE = config['MM1_MAX_REVERSE']

    def on_close(self):
        logger.info("Client disconnected")


class VideoAPI(RequestHandler):
    '''
    Serves a MJPEG of the images posted from the vehicle.
    '''

    async def get(self):
        placeholder_image = utils.load_image_sized(
                        os.path.join(self.application.static_file_path,
                                     "img_placeholder.jpg"), 160, 120, 3)

        self.set_header("Content-type",
                        "multipart/x-mixed-replace;boundary=--boundarydonotcross")

        served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\n"
        while True:

            interval = .005
            if served_image_timestamp + interval < time.time():
                #
                # if we have an image, then use it.
                # otherwise show placeholder
                #
                if hasattr(self.application, 'img_arr') and self.application.img_arr is not None:
                    img = utils.arr_to_binary(self.application.img_arr)
                else:
                    img = utils.arr_to_binary(placeholder_image)

                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(img)
                served_image_timestamp = time.time()
                try:
                    await self.flush()
                except tornado.iostream.StreamClosedError:
                    pass
            else:
                await tornado.gen.sleep(interval)


class WebRTCOfferAPI(RequestHandler):
    """Accept a browser WebRTC offer and return SDP answer."""

    @staticmethod
    def _build_rtc_configuration(ice_servers):
        if not WEBRTC_AVAILABLE:
            return None

        rtc_servers = []
        for item in ice_servers or []:
            if isinstance(item, str) and item:
                rtc_servers.append(RTCIceServer(urls=item))
                continue
            if isinstance(item, dict) and item.get('urls'):
                kwargs = {'urls': item['urls']}
                if item.get('username') is not None:
                    kwargs['username'] = item['username']
                if item.get('credential') is not None:
                    kwargs['credential'] = item['credential']
                rtc_servers.append(RTCIceServer(**kwargs))

        if not rtc_servers:
            return None
        return RTCConfiguration(iceServers=rtc_servers)

    async def post(self):
        if not WEBRTC_AVAILABLE:
            self.set_status(501)
            self.write({
                "error": "WebRTC dependencies missing. Install donkeycar[webrtc]."
            })
            return
        if not bool(getattr(self.application, 'webrtc_enabled', True)):
            self.set_status(403)
            self.write({"error": "WebRTC is disabled in configuration"})
            return

        payload = tornado.escape.json_decode(self.request.body)
        if not isinstance(payload, dict) or 'sdp' not in payload or 'type' not in payload:
            self.set_status(400)
            self.write({"error": "Invalid offer payload"})
            return

        rtc_config = self._build_rtc_configuration(
            getattr(self.application, 'webrtc_ice_servers', [])
        )
        if rtc_config is None:
            pc = RTCPeerConnection()
        else:
            pc = RTCPeerConnection(configuration=rtc_config)
        self.application.webrtc_peers.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self.application.webrtc_peers.discard(pc)

        pc.addTrack(DonkeyVideoStreamTrack(self.application))
        offer = RTCSessionDescription(sdp=payload['sdp'], type=payload['type'])
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        self.set_header("Content-Type", "application/json")
        self.write({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        })


class WebRTCConfigAPI(RequestHandler):
    """Expose browser WebRTC transport configuration."""

    def get(self):
        enabled = bool(getattr(self.application, 'webrtc_enabled', True))
        ice_servers = getattr(self.application, 'webrtc_ice_servers', []) or []
        self.set_header("Content-Type", "application/json")
        self.write({
            "enabled": enabled,
            "available": bool(WEBRTC_AVAILABLE),
            "iceServers": ice_servers,
        })


class WebRTCHealthAPI(RequestHandler):
    """Simple status endpoint to validate WebRTC runtime state."""

    def get(self):
        self.set_header("Content-Type", "application/json")
        self.write({
            "enabled": bool(getattr(self.application, 'webrtc_enabled', True)),
            "available": bool(WEBRTC_AVAILABLE),
            "activePeers": len(getattr(self.application, 'webrtc_peers', [])),
        })


class BaseHandler(RequestHandler):
    """ Serves the FPV web page"""
    async def get(self):
        data = {}
        await self.render("templates/base_fpv.html", **data)


class WebFpv(Application):
    """
    Class for running an FPV web server that only shows the camera in real-time.
    The web page contains the camera view and auto-adjusts to the web browser
    window size. Conjecture: this picture up-scaling is performed by the
    client OS using graphics acceleration. Hence a web browser on the PC is
    faster than a pure python application based on open cv or similar.
    """

    def __init__(self, port=8890):
        self.port = port
        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')

        """Construct and serve the tornado application."""
        handlers = [
            (r"/", BaseHandler),
            (r"/video", VideoAPI),
            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path})
        ]

        settings = {'debug': True}
        self.img_arr = None
        super().__init__(handlers, **settings)
        logger.info(f"Started Web FPV server. You can now go to "
                    f"{gethostname()}.local:{self.port} to view the car camera")

    def update(self):
        """ Start the tornado webserver. """
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.listen(self.port)
        IOLoop.instance().start()

    def run_threaded(self, img_arr=None):
        self.img_arr = img_arr

    def run(self, img_arr=None):
        self.img_arr = img_arr

    def shutdown(self):
        pass


