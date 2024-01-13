import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import easyocr
import re


reader = easyocr.Reader(['en'],gpu=True)


def bank_details(cropped_img):
    temp_path = 'temp_cropped_image.jpg'
    cv2.imwrite(temp_path,cropped_img)
    result = reader.readtext(temp_path)
    
    b=[]
    for j in result:
        b.append(j[1])
    
    c=' '.join(b)
    e=re.findall('^(.*?)(?=IFS)',c)
    bnk_nme_addrs=e[0]
    
    
    c=' '.join(b).upper().replace(' CODE','C').replace(' ','')
    pattern=re.compile('[^a-zA-Z0-9]')
    c1=pattern.sub('',c)
    ifsc_no=re.findall(r'IFSC(.{11})',c1)
    
    return bnk_nme_addrs,ifsc_no[0]

def accnt_no(cropped_img):
    temp_path = 'temp_cropped_image.jpg'
    cv2.imwrite(temp_path,cropped_img)
    result = reader.readtext(temp_path)
    
    b=[]
    for j in result:
        b.append(j[1])
        
    c=''.join(b).replace(' ','')
    e=re.findall('[0-9]*',c)
    acc_no=' '.join([i for i in e if i!='' ])
    return acc_no

def micr_strip(cropped_img):
    temp_path = 'temp_cropped_image.jpg'
    cv2.imwrite(temp_path,cropped_img)
    result = reader.readtext(temp_path)
    
    b=[]
    for j in result:
        b.append(j[1])
        
    c=''.join(b).replace(' ','')
    e=re.findall('[0-9]*',c)
    cheq_no=' '.join([i for i in e if i!='' ])

    return cheq_no

def main():
    st.set_page_config(layout="wide")
    st.title("Cheque Data Extraction App")
    
    col1, col2 = st.columns(2)
    bnk_add_nme=''
    ifsc_no=''
    acc_no=''
    cheq_no=''
    # Add content to the first column
    with col1:

        model_path=r"\model\weights\best.pt"
        model=YOLO(model_path)
        
        # File upload widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        print(type(uploaded_file))
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
        
        if uploaded_file is not None:
            pred_img=model.predict(source=image_array)
            a=pred_img[0].orig_img
            plt.figure(figsize=(25,25))
            img_org=cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

            images_data=[]
            for ind,i in enumerate(pred_img[0].boxes.xyxy):
                cls_no=int(pred_img[0].boxes.cls[ind])
                cls_name=pred_img[0].names[cls_no]
                cnf_scr=float(pred_img[0].boxes.conf[ind])
                
                x_min=int(i[0])
                y_min=int(i[1])
                x_max=int(i[2])
                y_max=int(i[3])
                cropped_img = img_org[y_min:y_max, x_min:x_max]
                images_data.append({'class':cls_name,'score':cnf_scr,'image':cropped_img})
                
                if cls_name=='Bank Details':
                    bnk_add_nme,ifsc_no=bank_details(cropped_img)
                elif cls_name=='Account Number':
                    acc_no=accnt_no(cropped_img)
                elif cls_name=='MICR Strip':
                    cheq_no=micr_strip(cropped_img)
                    

    # Add content to the second column
    with col2:

    # You can continue adding content outside of the columns layout manager
        if bnk_add_nme=='':
            bnk_add_nme=''
            ifsc_no=''
            acc_no=''
            cheq_no=''
            # st.write("This content is outside of the columns.")
            
        if bnk_add_nme!='':
            account_heading = f"<div style='padding: 10px; padding-top:20px; border-radius: 10px;'><h4 style='color: white; text-shadow: 2px 2px 2px #333;'>Bank Details</h4></div>"
            st.markdown(account_heading, unsafe_allow_html=True)

            # Display the bank account_no with chat-like background and embossing
            account_details = f"<div style='background-color: rgb(38, 39, 48); padding: 10px; max-width:70%; border-radius: 10px; box-shadow: 2px 2px 2px #888888;'><h7>{bnk_add_nme}</h7></div>"
            st.markdown(account_details, unsafe_allow_html=True)
            
            account_heading = f"<div style='padding: 10px;  margin-top:40px; border-radius: 10px;'><h4 style='color: white; text-shadow: 2px 2px 2px #333;'>IFSC Code</h4></div>"
            st.markdown(account_heading, unsafe_allow_html=True)

            # Display the bank account_no with chat-like background and embossing
            account_details = f"<div style='background-color: rgb(38, 39, 48); padding: 10px; max-width:70%; border-radius: 10px; box-shadow: 2px 2px 2px #888888;'><h7>{ifsc_no}</h7></div>"
            st.markdown(account_details, unsafe_allow_html=True)
            
            account_heading = f"<div style='padding: 10px; margin-top:40px; border-radius: 10px;'><h4 style='color: white; text-shadow: 2px 2px 2px #333;'>Account Number</h4></div>"
            st.markdown(account_heading, unsafe_allow_html=True)

            # Display the bank account_no with chat-like background and embossing
            account_details = f"<div style='background-color: rgb(38, 39, 48); padding: 10px; max-width:70%; border-radius: 10px; box-shadow: 2px 2px 2px #888888;'><h7>{acc_no}</ph7</div>"
            st.markdown(account_details, unsafe_allow_html=True)
            
            account_heading = f"<div style='padding: 10px; margin-top:40px; border-radius: 10px;'><h4 style='color: white; text-shadow: 2px 2px 2px #333;'>Cheque Number</h4></div>"
            st.markdown(account_heading, unsafe_allow_html=True)

            # Display the bank account_no with chat-like background and embossing
            account_details = f"<div style='background-color: rgb(38, 39, 48); padding: 10px; max-width:70%; border-radius: 10px; box-shadow: 2px 2px 2px #888888;'><h7>{cheq_no}</h7></div>"
            st.markdown(account_details, unsafe_allow_html=True)
        
if __name__ == "__main__":
    
    main()
