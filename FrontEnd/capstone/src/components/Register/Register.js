import React, { useState } from "react";
import axios from "axios";
import "./Register.css";

const Register = () => {
  const [user, setUser] = useState({
    nameregister: "",
    phn: "",
    gender: "",
    age: "",
    emailregister: "",
    user_name: "",
    pass: "",
    repass: "",
  });

  const handleChange = (event) => {
    console.log(user.nameregister);
    console.log(user.emailregister);
    console.log(user.age);

    console.log(user.phn);
    console.log(user.user_name);
    console.log(user.pass);
    console.log(user.repass);

    var name1 = event.target.id;
    var value1 = event.target.value;

    setUser({
      ...user,
      [name1]: value1,
    });

    if (value1 != "" && name1 == "repass") {
      var value2 = user.pass;
      if (value2 != value1) {
        document.getElementById("hov").style.display = "inline-block";
      } else {
        document.getElementById("hov").style.display = "none";
      }
    }
  };

  const handleSubmit = (e) => {
    console.log(user.nameregister);
    console.log(user.emailregister);

    console.log(user.phn);

    console.log(user.user_name);
    console.log(user.pass);
    console.log(user.repass);

    e.preventDefault();

    var checkedradio = document.querySelector('input[name="gender"]:checked');
    console.log(checkedradio.id);
    if (checkedradio.id == "radio1") {
      user.gender = "MALE";
    } else if (checkedradio.id == "radio2") {
      user.gender = "FEMALE";
    }
    if (
      user.nameregister != "" &&
      user.emailregister != "" &&
      user.phn != "" &&
      user.user_name != "" &&
      user.pass != "" &&
      user.gender != "" &&
      user.repass != "" &&
      user.age != ""
    ) {
      if (true) {
        let headers = new Headers();

        headers.append("Content-Type", "application/json");
        headers.append("Accept", "application/json");

        headers.append("Origin", "http://localhost:1086");
        headers.append("Access-Control-Allow-Origin", "*");

        const url = `http://localhost:9995/register?type=register&firstone=fist&name=${user.nameregister}&gender=${user.gender}&phn=${user.phn}&email=${user.emailregister}&pass=${user.pass}&repass=${user.repass}&lastest=ooop&age=${user.age}&user_name=${user.user_name}&lastone=last`;
        axios
          .post(url, user)
          .then((response) => {
            console.log(response.status);
            if (response.status == 200) {
              alert("You have registered successfully");
              window.location.href = "loginpage";
            } else if (response.status == 263) {
              alert("Username is already taken. Please choose another one!");
            }
          })
          .then((html) => console.log(html));
      } else {
        alert("Re-Entered Password Doesn't Match the Password..");
      }
    } else {
      alert("Empty Feilds Are Not Allowed..");
    }
  };

  return (
    <div className="bodyy">
      <br />
      <div id="toppr">
        <div id="brandname2">
          <h2>Summcreat</h2>
        </div>
        <div id="tagline2">
          <h4>Create Your Summaries now!</h4>
        </div>
      </div>
      <div className="containerrr">
        <div className="h111">
          <br />
          <h1>CREATE ACCOUNT</h1>
        </div>
        <hr width="75%" className="hrrr" />
        <input
          type="text"
          placeholder="Your Name"
          className="inp_styles"
          id="nameregister"
          onChange={handleChange}
        />
        <input
          type="text"
          placeholder="Your Email"
          className="inp_styles"
          id="emailregister"
          onChange={handleChange}
        />
        <input
          type="text"
          placeholder="Your Phone Number"
          className="inp_styles"
          id="phn"
          onChange={handleChange}
        />
        <input
          type="text"
          placeholder="Your Age"
          className="inp_styles"
          id="age"
          onChange={handleChange}
        />

        <br />
        <div id="genderdiv">
          <label htmlfor="gender" id="genderlabel">
            Gender:
          </label>
          <input
            type="radio"
            value="Male"
            name="gender"
            id="radio1"
            className="check_styles"
          />
          <label htmlfor="gender" id="labels">
            Male
          </label>
          <input
            type="radio"
            id="radio2"
            name="gender"
            className="check_styles"
            value="Female"
          />
          <label htmlfor="gender" id="labels">
            Female
          </label>
        </div>
        <br />

        <input
          type="text"
          placeholder="Your Username"
          className="inp_styles"
          id="user_name"
          onChange={handleChange}
        />
        <input
          type="password"
          placeholder="Your Password"
          className="inp_styles"
          id="pass"
          onChange={handleChange}
        />
        <input
          type="password"
          placeholder="Repeat Your Password"
          className="inp_styles"
          id="repass"
          onChange={handleChange}
        />
        <label for="hov" id="hov" style={{ display: "none" }}>
          Re-entered Password Doesn't match
        </label>
        <input
          type="button"
          value="Register"
          className="but_style"
          id="btn1"
          onClick={handleSubmit}
        />
        <br />
        <div className="h44">
          <h4>
            Already have an account? <a href="/loginpage">Login Here</a>
          </h4>
        </div>
        <br />
        <br />
      </div>
      <br />
      <br />
      <br />
    </div>
  );
};

export default Register;
