from util.capture_user_video import CaptureVideo
from util.extract_keyponts import ExtractKeypoints
from util.prediction import Predict
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__=="__main__":
    logger.info("Implementation Started...")
    capture = Capt