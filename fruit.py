import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import random
import os

# Load images
imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/gameOver.png")

# Load fruit images from the 'fruitImages' folder under 'Resources'
fruitImages = []
fruitFolder = "Resources/fruitImages/"
for i in range(2, 16):  # Assuming 15 fruit images named fruit1.png, fruit2.png, ..., fruit15.png
    fruitPath = os.path.join(fruitFolder, f"fruits{i}.png")
    if os.path.exists(fruitPath):
        fruitImages.append(cv2.imread(fruitPath, cv2.IMREAD_UNCHANGED))

    # Ensure alpha channel for basket
imgBasket = cv2.imread("Resources/Basket.png", cv2.IMREAD_UNCHANGED)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initialize game variables
score = 0
basketY = 500  # Basket position
gameOver = False
fruits = []  # List of fruits
fruitSpeed = 5  # Initial fruit speed
maxSpeed = 20  # Cap the maximum speed
spawnNewFruit = True  # Controls whether a new fruit should spawn or not

# Initialize the webcam
cap = cv2.VideoCapture(0)


# Function to spawn a new fruit
def spawn_new_fruit():
    return {"image": random.choice(fruitImages), "pos": [random.randint(100, 900), 0], "speed": fruitSpeed}


# Spawn the first fruit
fruits.append(spawn_new_fruit())

while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.flip(img, 1)  # Flip horizontally for a mirror effect
    imgRaw = img.copy()

    # Resize background to match the webcam frame size
    imgBackgroundResized = cv2.resize(imgBackground, (img.shape[1], img.shape[0]))

    # Detect hands
    hands, img = detector.findHands(img, flipType=False)

    # Overlay the background
    img = cv2.addWeighted(img, 0.2, imgBackgroundResized, 0.8, 0)

    # Check for detected hands
    basketCenterX = None
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            basketCenterX = x + w // 2
            h1, w1, _ = imgBasket.shape
            basketX = basketCenterX - w1 // 2

            # Overlay the basket image
            img = cvzone.overlayPNG(img, imgBasket, (basketX, basketY))

            # Draw a red line at the bottom of the frame
    cv2.line(img, (0, img.shape[0] - 50), (img.shape[1], img.shape[0] - 50), (0, 0, 255), 5)

    # Game Over condition (fruit crosses the red line)
    for fruit in fruits[:]:  # Loop over a copy of the fruits list to avoid modifying it while iterating
        if fruit["pos"][1] > img.shape[0] - 50:
            gameOver = True
            fruits.remove(fruit)  # Remove fruit if it crosses the red line

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX, 2.5, (200, 0, 200), 5)
    else:
        # Move fruits downwards
        for fruit in fruits[:]:  # Again loop over a copy of the list
            fruit["pos"][1] += fruit["speed"]
            fruitPosInt = [int(fruit["pos"][0]), int(fruit["pos"][1])]
            img = cvzone.overlayPNG(img, fruit["image"], fruitPosInt)

            # Check for collision with basket
            if basketCenterX is not None:
                h1, w1, _ = imgBasket.shape
                basketLeft = basketCenterX - w1 // 2
                basketRight = basketCenterX + w1 // 2
                basketTop = basketY
                basketBottom = basketY + h1

                fruitCenterX = fruit["pos"][0] + fruit["image"].shape[1] // 2
                fruitCenterY = fruit["pos"][1] + fruit["image"].shape[0] // 2

                # Check if the fruit is within the basket boundaries
                if basketLeft <= fruitCenterX <= basketRight and basketTop <= fruitCenterY <= basketBottom:
                    score += 1
                    fruitSpeed = min(fruitSpeed + 1, maxSpeed)  # Increase speed with a cap
                    fruits.remove(fruit)  # Remove caught fruit
                    spawnNewFruit = True  # Allow spawning of new fruit
                    break

                    # Spawn a new fruit when the first fruit crosses 1/4th of the screen height
        for fruit in fruits:
            if fruit["pos"][1] > img.shape[0] // 4 and len(
                    fruits) == 1:  # If only one fruit exists and it has crossed 1/4th of the screen height
                fruits.append(spawn_new_fruit())  # Spawn a new fruit after the first fruit crosses 1/4th of the path

        # Display score
        cv2.putText(img, f"Score: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

        # Show the raw webcam feed in the bottom corner
        img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

        # Display the final image
    cv2.imshow("Image", img)

    # Key press events
    key = cv2.waitKey(1)
    if key == ord('r'):  # Reset game
        fruits = [spawn_new_fruit()]
        gameOver = False
        score = 0
        fruitSpeed = 5  # Reset fruit speed
        imgGameOver = cv2.imread("Resources/gameOver.png")
    elif key == ord('q'):  # Quit game
        break

    # Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
