"""UniDepth-KD: Knowledge Distillation for Monocular Depth Estimation using Vision Transformers"""

__version__ = "1.0.0"
__author__ = "Depth Estimation Team"

from .models.student_model import StudentModel, create_student_model
from .models.teacher_model import TeacherModel, create_teacher_model
from .models.vit_encoder import ViTEncoder, create_vit_encoder
from .models.cnn_encoder import CNNEncoder, create_cnn_encoder

__all__ = [
    'StudentModel',
    'TeacherModel',
    'ViTEncoder',
    'CNNEncoder',
    'create_student_model',
    'create_teacher_model',
    'create_vit_encoder',
    'create_cnn_encoder',
]
