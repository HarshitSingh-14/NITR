import React, { useContext, useRef, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { UserContext } from "./../App";
import M from "materialize-css";
export default function Navbar() {
  const Navigate = useNavigate();
  const [search, setSearch] = useState("");
  const { state, dispatch } = useContext(UserContext);
  const searchModal = useRef(null);
  const [userDetails, setUserDetails] = useState([]);
  useEffect(() => {
    M.Modal.init(searchModal.current);
  }, []);
  const renderlist = () => {
    if (state) {
      return [
        <li key="1">
          <i
            data-target="modal1"
            className="large material-icons modal-trigger"
            style={{ color: "black" }}
          >
            search
          </i>
        </li>,
        <li key="2">
          <Link to="/profile">Profile</Link>
        </li>,
        // <li key="3">
        //   <Link to="/create">Create Post</Link>
        // </li>,
        // <li key="4">
        //   <Link to="/myfollowingspost">My Following post</Link>
        // </li>,
        <li key="5">
          <button
            className="btn waves-effect waves-light  blue  darken-1"
            onClick={() => {
              localStorage.clear();
              dispatch({ type: "CLEAR" });
              Navigate("/signin");
            }}
          >
            Logout
          </button>
        </li>,
      ];
    } else {
      return [
        <li key="6">
          <Link to="/signin">Signin</Link>
        </li>,
        <li key="7">
          <Link to="/signup">Signup</Link>
        </li>,
      ];
    }
  };
  const fetchUsers = (query) => {
    setSearch(query);
    fetch("/search-users", {
      method: "post",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
      }),
    })
      .then((res) => res.json())
      .then((results) => {
        // console.log(results);
        setUserDetails(results.user);
      });
  };
  return (
    <div>
      <nav>
        <div className="nav-wrapper white">
          <Link to={state ? "/" : "/signin"} className="brand-logo left">
            Dr. cleaner
          </Link>
          <ul id="nav-mobile" className="right hide-on-med-and-down">
            {renderlist()}
            {/* <li>
              <Link to="/profile">Profile</Link>
            </li>
            <li>
              <Link to="/create">Create Post</Link>
            </li>
            <li>
              <Link to="/signin">Signin</Link>
            </li>
            <li>
              <Link to="/signup">Signup</Link>
            </li> */}
            <div
              id="modal1"
              className="modal"
              ref={searchModal}
              style={{ color: "black" }}
            >
              <div className="modal-content" style={{ color: "black" }}>
                <input
                  type="text"
                  placeholder="search users"
                  value={search}
                  onChange={(e) => {
                    // setSearch(e.target.value);
                    fetchUsers(e.target.value);
                  }}
                />
                <ul className="collection">
                  {userDetails !== null &&
                    userDetails.map((item) => {
                      return (
                        <Link
                          to={
                            item._id !== state._id
                              ? "/profile/" + item._id
                              : "/profile"
                          }
                          onClick={() => {
                            M.Modal.getInstance(searchModal.current).close();
                            setSearch("");
                          }}
                        >
                          <li className="collection-item">{item.email}</li>
                        </Link>
                      );
                    })}
                </ul>
              </div>
              <div className="modal-footer">
                <button
                  className="modal-close waves-effect waves-green btn-flat"
                  onClick={() => setSearch("")}
                >
                  close
                </button>
              </div>
            </div>
          </ul>
        </div>
      </nav>
    </div>
  );
}
