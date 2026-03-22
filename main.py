import argparse
from ad_inserter import AdPlacementSystem

def main():
    parser = argparse.ArgumentParser(description="In-Video Ad Placement System")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--logo", type=str, required=True, help="Path to input brand logo image")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Path to output processed video")
    parser.add_argument("--fps", type=int, default=None, help="Target FPS (optional)")

    args = parser.parse_args()

    system = AdPlacementSystem(target_fps=args.fps)
    system.process_video(args.video, args.logo, args.output)

if __name__ == "__main__":
    main()
