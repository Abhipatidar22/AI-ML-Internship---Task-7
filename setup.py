import subprocess
import sys

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False
    return True

def check_installation():
    """Check if all required packages are installed"""
    required_packages = [
        "sklearn",
        "numpy",
        "matplotlib",
        "pandas",
        "seaborn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        return False
    else:
        print("All required packages are installed!")
        return True

def main():
    print("SVM Classification Setup")
    print("=" * 25)
    
    if not check_installation():
        print("Installing missing packages...")
        if install_requirements():
            print("Setup completed successfully!")
            print("You can now run: python svm_classification.py")
        else:
            print("Setup failed. Please install packages manually.")
    else:
        print("Environment is ready!")
        print("You can run: python svm_classification.py")

if __name__ == "__main__":
    main()