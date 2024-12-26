import { useState } from 'react'
// Removed missing imports:
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
// import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        {/* Removed logo images */}
        {/* <img src={viteLogo} className="logo" alt="Vite logo" /> */}
        {/* <img src={reactLogo} className="logo react" alt="React logo" /> */}
      </div>
      <h1>Vite + React</h1>
      <div>
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
      </div>
    </>
  )
}

export default App
