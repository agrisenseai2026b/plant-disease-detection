import sys
import json
import os

from disease_pred import predict_disease


# ============================================================
# GET IMAGE PATH
# ============================================================

def get_image_path():

    # --------------------------------------------------------
    # OPTION 1:
    # Image path supplied as command-line argument
    # --------------------------------------------------------

    if len(sys.argv) >= 2:
        return sys.argv[1]

    # --------------------------------------------------------
    # OPTION 2:
    # Image path supplied through JSON stdin
    # Used by Node.js website
    # --------------------------------------------------------

    input_data = sys.stdin.read().strip()

    if not input_data:
        return None

    try:
        data = json.loads(input_data)

    except json.JSONDecodeError:
        return None

    return (
        data.get("image_path")
        or data.get("file_path")
        or data.get("filePath")
        or data.get("path")
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Get image path
        # ----------------------------------------------------

        image_path = get_image_path()

        if not image_path:

            print(
                json.dumps({
                    "success": False,
                    "error": "No image path was provided."
                })
            )

            sys.exit(1)

        # ----------------------------------------------------
        # Convert to absolute path
        # ----------------------------------------------------

        image_path = os.path.abspath(image_path)

        # ----------------------------------------------------
        # Check image
        # ----------------------------------------------------

        if not os.path.isfile(image_path):

            print(
                json.dumps({
                    "success": False,
                    "error": "Image file not found: " + image_path
                })
            )

            sys.exit(1)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = predict_disease(image_path)

        # ----------------------------------------------------
        # stdout MUST contain only JSON
        # ----------------------------------------------------

        print(
            json.dumps(result)
        )

    except Exception as e:

        print(
            json.dumps({
                "success": False,
                "error": str(e)
            })
        )

        sys.exit(1)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()