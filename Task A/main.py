
def print_menu():
    print("Gender Prediction")
    print("1. Train model")
    print("2. Predict single image")
    print("3. Predict batch folder")
    print("4. Exit")

def main():
    while True:
        print_menu()
        choice = input("Choose an option (1/2/3/4): ").strip()
        if choice == '1':
            from train import train_model
            train_model()
        elif choice == '2':
            from predict import predict_gender
            img_path = input("Enter image path: ").strip()
            predict_gender(img_path)
        elif choice == '3':
            from predict import batch_predict_gender
            folder_path = input("Enter folder path containing images: ").strip()
            batch_predict_gender(folder_path)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

# Required to run main() when script is executed
if __name__ == '__main__':
    main()
