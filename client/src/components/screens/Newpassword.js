import React, { useState, useContext } from "react";
import { UserContext } from "./../../App";
import { Link, useNavigate, useParams } from "react-router-dom";
import M from "materialize-css";
export default function Newpassword() {
  // console.log(UserContext);
  const Navigate = useNavigate();
  const [password, setPassword] = useState("");
  const { token } = useParams();
  //   console.log(token);
  //Cors or cross origin resource sharing has to be used for backend and frontend as frontend is on 3000 and backend is at 5000 ...
  //well now we have something better
  const PostData = () => {
    fetch("/new-password", {
      method: "post",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        password,
        token,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        if (data.error) {
          M.toast({
            html: data.error,
            classes: "toast-container red darken-3",
          });
        } else {
          M.toast({
            html: data.message,
            classes: "toast-container green darken-1",
          });
          Navigate("/signin");
        }
      })
      .catch((err) => {
        console.log(err);
      });
  };
  return (
    <div className="mycard">
      <div className="card auth-card input-field">
        <h2>Instagram</h2>
        <input
          type="password"
          placeholder="new password"
          value={password}
          onChange={(e) => {
            setPassword(e.target.value);
          }}
        />
        <button
          className="btn waves-effect waves-light  blue  darken-1"
          onClick={() => PostData()}
        >
          update password
        </button>
      </div>
    </div>
  );
}
