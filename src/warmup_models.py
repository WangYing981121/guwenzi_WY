#!/usr/bin/env python3
from paddleocr import PaddleOCR


def main():
    PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        use_gpu=False,
        show_log=False,
    )
    print("PaddleOCR models are ready.")


if __name__ == "__main__":
    main()
