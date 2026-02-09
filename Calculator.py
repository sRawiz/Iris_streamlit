import streamlit as st

st.set_page_config(page_title="Calculator", layout="centered")

st.title("Calculator")

col1, col2 = st.columns(2)

with col1:
    num1 = st.number_input("Enter first number:", value=0.0, format="%.2f")

with col2:
    num2 = st.number_input("Enter second number:", value=0.0, format="%.2f")

operation = st.selectbox(
    "Select operation:",
    ["Add (+)", "Subtract (-)", "Multiply (×)", "Divide (÷)"]
)

if st.button("Calculate", type="primary"):
    result = None
    
    if operation == "Add (+)":
        result = num1 + num2
        st.success(f"**Result:** {num1} + {num2} = **{result}**")
    
    elif operation == "Subtract (-)":
        result = num1 - num2
        st.success(f"**Result:** {num1} - {num2} = **{result}**")
    
    elif operation == "Multiply (×)":
        result = num1 * num2
        st.success(f"**Result:** {num1} × {num2} = **{result}**")
    
    elif operation == "Divide (÷)":
        if num2 != 0:
            result = num1 / num2
            st.success(f"**Result:** {num1} ÷ {num2} = **{result}**")
        else:
            st.error("Error: Cannot divide by zero!")

st.markdown("---")
st.markdown("Rawich Chanchad")