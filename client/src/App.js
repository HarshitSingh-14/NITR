import React, { createContext, useReducer, useEffect, useContext } from "react";
import Navbar from "./components/Navbar";
import "./App.css";
import { BrowserRouter, Routes, Route, useNavigate } from "react-router-dom";
import Home from "./components/screens/Home";
import Signin from "./components/screens/Signin";
import Signup from "./components/screens/Signup";
import Profile from "./components/screens/Profile";
import CreatePost from "./components/screens/CreatePost";
import { reducer, initialState } from "./reducers/userReducer";
import UserProfile from "./components/screens/UserProfile";
import Reset from "./components/screens/Reset";
import SubscribesUserPosts from "./components/screens/SubscribesUserPosts";
import Newpassword from "./components/screens/Newpassword";
import Drone from "./components/screens/Drone";
export const UserContext = createContext();
const Routing = () => {
  const Navigate = useNavigate();
  // console.log(UserContext);
  const { state, dispatch } = useContext(UserContext);
  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("user_NITR"));
    if (user) {
      dispatch({ type: "USER", payload: user });
      // Navigate("/");
    } else {
      if (!window.location.pathname.startsWith("/reset")) {
        console.log("");
        Navigate("/signin");
      }
    }
  }, []);
  return (
    <Routes>
      {/* <Route exact path="/" element={<Home />}></Route> */}
      <Route exact path="/" element={<Drone />}></Route>
      <Route exact path="/signin" element={<Signin />}></Route>
      <Route exact path="/signup" element={<Signup />}></Route>
      <Route exact path="/profile" element={<Profile />}></Route>
      <Route exact path="/create" element={<CreatePost />}></Route>
      <Route exact path="/profile/:user_id" element={<UserProfile />}></Route>
      <Route exact path="/reset" element={<Reset />}></Route>
      <Route exact path="/reset/:token" element={<Newpassword />}></Route>
      <Route exact path="/drone" element={<Drone />}></Route>
      <Route
        exact
        path="/myfollowingspost"
        element={<SubscribesUserPosts />}
      ></Route>
    </Routes>
  );
};
function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <UserContext.Provider value={{ state, dispatch }}>
      <BrowserRouter>
        <div>
          <Navbar />
          <Routing />
        </div>
      </BrowserRouter>
    </UserContext.Provider>
  );
}

export default App;
