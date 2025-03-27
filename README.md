# IMAGE-TRANSFORMATIONS

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the required packages.

### Step2:
Load the image file in the program.

### Step3:
Use the techniques for Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

### Step4:
Display the modified image output

### Step5:
End the program.

## Program:
```python
Developed By : Dharshni V M 
Register Number : 212223240029

i)Image Translation

import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image=cv2.imread("kakashi.png")
input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
print("Dharshni V M \n212223240029")
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
rows,cols,dim=input_image.shape
M=np.float32([[1,0,100],[0,1,200],[0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(cols,rows))
plt.subplot(1, 2, 2)
plt.imshow(translated_image)
plt.axis('off')
plt.title("Image Translation")
plt.show()

ii) Image Scaling

input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
print("Dharshni V M \n212223240029")
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
rows,cols,dim=input_image.shape
M=np.float32([[1.5,0,0],[0,1.8,0],[0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(cols*2,rows*2))
plt.subplot(1, 2, 2)
plt.imshow(translated_image)
plt.axis('off')
plt.title("Image Scaling")
plt.show()

iii)Image shearing

input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
print("Dharshni V M \n212223240029")
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
rows,cols,dim=input_image.shape
M1=np.float32([[1,0.5,0],[0,1,0],[0,0,1]])
M2=np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
translated_image1=cv2.warpPerspective(input_image,M1,(int(cols*1.5),int(rows*1.5)))
translated_image2=cv2.warpPerspective(input_image,M2,(int(cols*1.5),int(rows*1.5)))
plt.subplot(1, 2, 2)
plt.imshow(translated_image1)
plt.imshow(translated_image2)
plt.axis('off')
plt.title("Image Shearing")
plt.show()

iv)Image Reflection

input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
print("Dharshni V M \n212223240029")
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
rows,cols,dim=input_image.shape
M1=np.float32([[1,0,0],[0,-1,rows],[0,0,1]])
M2=np.float32([[-1,0,cols],[0,1,0],[0,0,1]])
translated_image1=cv2.warpPerspective(input_image,M1,(int(cols),int(rows)))
translated_image2=cv2.warpPerspective(input_image,M2,(int(cols),int(rows)))
plt.subplot(1, 2, 2)
plt.imshow(translated_image1)
plt.imshow(translated_image2)
plt.axis('off')
plt.title("Image Reflection")
plt.show()

v)Image Rotation

input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
print("Dharshni V M \n212223240029")
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
rows,cols,dim=input_image.shape
angle=np.radians(10)
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(int(cols),int(rows)))
plt.subplot(1, 2, 2)
plt.imshow(translated_image)
plt.axis('off')
plt.title("Image Rotation")
plt.show()

vi)Image Cropping

h, w, _ = input_image.shape
cropped_face = input_image[int(h*0.2):int(h*0.8), int(w*0.3):int(w*0.7)]
cv2.imwrite("cropped_face.png", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
print("Dharshni V M \n212223240029")
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
plt.subplot(1, 2, 2)
plt.imshow(cropped_face)  
plt.axis('off')
plt.title("Cropped Image")
plt.show()

```
## Output:
### i)Image Translation

![Translation](https://github.com/user-attachments/assets/936e361a-056c-401f-8607-affc942a4477)

### ii) Image Scaling

![scaling](https://github.com/user-attachments/assets/5186f2cf-0f25-4cc1-9d37-7dbca877cbb3)

### iii)Image shearing

![shearing](https://github.com/user-attachments/assets/d36fbfb6-a49c-4391-9989-abfb0c4f0ca6)

### iv)Image Reflection

![reflection](https://github.com/user-attachments/assets/af1e9361-2be9-44d8-9c98-2cf5fbe86245)

### v)Image Rotation

![rotation](https://github.com/user-attachments/assets/bf3e4c73-7894-4d5f-ba34-58e89a515401)

### vi)Image Cropping

![cropped](https://github.com/user-attachments/assets/fc4215b0-a6fc-4dc1-891a-2ec0bb071864)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
