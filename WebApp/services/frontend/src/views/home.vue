<template>
  <div class="home-container">
    <div class="home-header">
      <img
        alt="image"
        src="/external/fh_wedel_logo.svg-200h.png"
        class="home-image"
      />
      <header data-thq="thq-navbar" class="home-navbar-interactive">
        <div class="home-container01">
          <span class="home-logo">Alles auf einen Blick!</span>
        </div>
        <svg viewBox="0 0 1024 1024" class="home-icon">
          <path
            d="M682 726v-256q0-82-46-137t-124-55-124 55-46 137v256h340zM768 682l86 86v42h-684v-42l86-86v-212q0-100 51-174t141-96v-30q0-26 18-45t46-19 46 19 18 45v30q90 22 141 96t51 174v212zM512 938q-34 0-60-25t-26-59h172q0 34-26 59t-60 25z"
          ></path>
        </svg>
        <div class="home-container02">
          <div class="home-profile">
            <div class="home-container03">
              <img alt="image" src="/profil-200h.jpg" class="home-image1" />
            </div>
            <div class="home-container04">
              <span class="home-text">Thomas</span>
              <span class="home-text01">View Profile</span>
            </div>
          </div>
          <div data-thq="thq-navbar-nav" class="home-desktop-menu"></div>
        </div>
        <div data-thq="thq-burger-menu" class="home-burger-menu">
          <svg viewBox="0 0 1024 1024" class="home-icon02">
            <path
              d="M128 554.667h768c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-768c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667zM128 298.667h768c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-768c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667zM128 810.667h768c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-768c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667z"
            ></path>
          </svg>
        </div>
        <div data-thq="thq-mobile-menu" class="home-mobile-menu">
          <div class="home-nav">
            <div class="home-top">
              <span class="home-logo1">STOCKAI</span>
              <div data-thq="thq-close-menu" class="home-close-menu">
                <svg viewBox="0 0 1024 1024" class="home-icon04">
                  <path
                    d="M810 274l-238 238 238 238-60 60-238-238-238 238-60-60 238-238-238-238 60-60 238 238 238-238z"
                  ></path>
                </svg>
              </div>
            </div>
            <nav class="home-links">
              <span class="home-nav1">About</span>
              <span class="home-nav2">Features</span>
              <span class="home-nav3">Pricing</span>
              <span class="home-nav4">Team</span>
              <span class="home-nav5">Blog</span>
            </nav>
            <div class="home-buttons">
              <button class="home-login button">Login</button>
              <button class="home-register button">Register</button>
            </div>
          </div>
          <div>
            <svg viewBox="0 0 950.8571428571428 1024" class="home-icon06">
              <path
                d="M925.714 233.143c-25.143 36.571-56.571 69.143-92.571 95.429 0.571 8 0.571 16 0.571 24 0 244-185.714 525.143-525.143 525.143-104.571 0-201.714-30.286-283.429-82.857 14.857 1.714 29.143 2.286 44.571 2.286 86.286 0 165.714-29.143 229.143-78.857-81.143-1.714-149.143-54.857-172.571-128 11.429 1.714 22.857 2.857 34.857 2.857 16.571 0 33.143-2.286 48.571-6.286-84.571-17.143-148-91.429-148-181.143v-2.286c24.571 13.714 53.143 22.286 83.429 23.429-49.714-33.143-82.286-89.714-82.286-153.714 0-34.286 9.143-65.714 25.143-93.143 90.857 112 227.429 185.143 380.571 193.143-2.857-13.714-4.571-28-4.571-42.286 0-101.714 82.286-184.571 184.571-184.571 53.143 0 101.143 22.286 134.857 58.286 41.714-8 81.714-23.429 117.143-44.571-13.714 42.857-42.857 78.857-81.143 101.714 37.143-4 73.143-14.286 106.286-28.571z"
              ></path></svg
            ><svg viewBox="0 0 877.7142857142857 1024" class="home-icon08">
              <path
                d="M585.143 512c0-80.571-65.714-146.286-146.286-146.286s-146.286 65.714-146.286 146.286 65.714 146.286 146.286 146.286 146.286-65.714 146.286-146.286zM664 512c0 124.571-100.571 225.143-225.143 225.143s-225.143-100.571-225.143-225.143 100.571-225.143 225.143-225.143 225.143 100.571 225.143 225.143zM725.714 277.714c0 29.143-23.429 52.571-52.571 52.571s-52.571-23.429-52.571-52.571 23.429-52.571 52.571-52.571 52.571 23.429 52.571 52.571zM438.857 152c-64 0-201.143-5.143-258.857 17.714-20 8-34.857 17.714-50.286 33.143s-25.143 30.286-33.143 50.286c-22.857 57.714-17.714 194.857-17.714 258.857s-5.143 201.143 17.714 258.857c8 20 17.714 34.857 33.143 50.286s30.286 25.143 50.286 33.143c57.714 22.857 194.857 17.714 258.857 17.714s201.143 5.143 258.857-17.714c20-8 34.857-17.714 50.286-33.143s25.143-30.286 33.143-50.286c22.857-57.714 17.714-194.857 17.714-258.857s5.143-201.143-17.714-258.857c-8-20-17.714-34.857-33.143-50.286s-30.286-25.143-50.286-33.143c-57.714-22.857-194.857-17.714-258.857-17.714zM877.714 512c0 60.571 0.571 120.571-2.857 181.143-3.429 70.286-19.429 132.571-70.857 184s-113.714 67.429-184 70.857c-60.571 3.429-120.571 2.857-181.143 2.857s-120.571 0.571-181.143-2.857c-70.286-3.429-132.571-19.429-184-70.857s-67.429-113.714-70.857-184c-3.429-60.571-2.857-120.571-2.857-181.143s-0.571-120.571 2.857-181.143c3.429-70.286 19.429-132.571 70.857-184s113.714-67.429 184-70.857c60.571-3.429 120.571-2.857 181.143-2.857s120.571-0.571 181.143 2.857c70.286 3.429 132.571 19.429 184 70.857s67.429 113.714 70.857 184c3.429 60.571 2.857 120.571 2.857 181.143z"
              ></path></svg
            ><svg viewBox="0 0 602.2582857142856 1024" class="home-icon10">
              <path
                d="M548 6.857v150.857h-89.714c-70.286 0-83.429 33.714-83.429 82.286v108h167.429l-22.286 169.143h-145.143v433.714h-174.857v-433.714h-145.714v-169.143h145.714v-124.571c0-144.571 88.571-223.429 217.714-223.429 61.714 0 114.857 4.571 130.286 6.857z"
              ></path>
            </svg>
          </div>
        </div>
      </header>
    </div>
    <div class="home-container05">
      <div class="home-container06">
        <div class="home-sidebar">
          <div class="home-container07">
            <div class="home-container08">
              <div class="home-container09">
                <svg viewBox="0 0 1024 1024" class="home-icon12">
                  <path
                    d="M128 554.667h768c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-768c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667zM128 298.667h768c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-768c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667zM128 810.667h768c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-768c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667z"
                  ></path>
                </svg>
                <span class="home-text02">Dashboard</span>
              </div>
              <div class="home-container10">
                <svg viewBox="0 0 1024 1024" class="home-icon14">
                  <path
                    d="M86 682l170-170v128h640v86h-640v128zM938 342l-170 170v-128h-640v-86h640v-128z"
                  ></path>
                </svg>
                <span class="home-text03">Transaktionen</span>
              </div>
              <div class="home-container11">
                <svg viewBox="0 0 1024 1024" class="home-icon16">
                  <path
                    d="M640 790q102 0 180-68l76 76q-108 98-256 98-124 0-223-72t-139-184h-192l42-86h130q-2-12-2-42t2-42h-172l42-86h150q40-112 139-184t223-72q60 0 136 29t120 69l-76 76q-78-68-180-68-70 0-142 44t-104 106h288l-42 86h-274q-4 28-4 42t4 42h316l-42 86h-246q32 62 104 106t142 44z"
                  ></path>
                </svg>
                <span class="home-text04">Guthaben</span>
              </div>
              <div class="home-container12">
                <svg viewBox="0 0 1024 1024" class="home-icon18">
                  <path
                    d="M832 128q34 0 60 26t26 60v596q0 34-26 60t-60 26h-640q-34 0-60-26t-26-60v-596q0-34 26-60t60-26h640zM832 814v-600h-640v600h640zM726 726h-86v-172h86v172zM554 726h-84v-428h84v428zM384 726h-86v-300h86v300z"
                  ></path>
                </svg>
                <router-link to="/statistik" class="home-navlink">
                  Statistiken
                </router-link>
              </div>
              <div class="home-container13">
                <svg viewBox="0 0 1024 1024" class="home-icon20">
                  <path
                    d="M512 598q108 0 225 47t117 123v86h-684v-86q0-76 117-123t225-47zM512 512q-70 0-120-50t-50-120 50-121 120-51 120 51 50 121-50 120-120 50z"
                  ></path>
                </svg>
                <span class="home-text05">Persönlich</span>
              </div>
              <div class="home-container14">
                <div class="home-container15">
                  <svg viewBox="0 0 1024 1024" class="home-icon22">
                    <path
                      d="M768 342v-86h-512v86h512zM598 598v-86h-342v86h342zM256 384v86h512v-86h-512zM854 86q34 0 59 25t25 59v512q0 34-25 60t-59 26h-598l-170 170v-768q0-34 25-59t59-25h684z"
                    ></path>
                  </svg>
                </div>
                <span class="home-text06">Nachrichten</span>
              </div>
              <div class="home-container16">
                <svg viewBox="0 0 877.7142857142857 1024" class="home-icon24">
                  <path
                    d="M585.143 512c0-80.571-65.714-146.286-146.286-146.286s-146.286 65.714-146.286 146.286 65.714 146.286 146.286 146.286 146.286-65.714 146.286-146.286zM877.714 449.714v126.857c0 8.571-6.857 18.857-16 20.571l-105.714 16c-6.286 18.286-13.143 35.429-22.286 52 19.429 28 40 53.143 61.143 78.857 3.429 4 5.714 9.143 5.714 14.286s-1.714 9.143-5.143 13.143c-13.714 18.286-90.857 102.286-110.286 102.286-5.143 0-10.286-2.286-14.857-5.143l-78.857-61.714c-16.571 8.571-34.286 16-52 21.714-4 34.857-7.429 72-16.571 106.286-2.286 9.143-10.286 16-20.571 16h-126.857c-10.286 0-19.429-7.429-20.571-17.143l-16-105.143c-17.714-5.714-34.857-12.571-51.429-21.143l-80.571 61.143c-4 3.429-9.143 5.143-14.286 5.143s-10.286-2.286-14.286-6.286c-30.286-27.429-70.286-62.857-94.286-96-2.857-4-4-8.571-4-13.143 0-5.143 1.714-9.143 4.571-13.143 19.429-26.286 40.571-51.429 60-78.286-9.714-18.286-17.714-37.143-23.429-56.571l-104.571-15.429c-9.714-1.714-16.571-10.857-16.571-20.571v-126.857c0-8.571 6.857-18.857 15.429-20.571l106.286-16c5.714-18.286 13.143-35.429 22.286-52.571-19.429-27.429-40-53.143-61.143-78.857-3.429-4-5.714-8.571-5.714-13.714s2.286-9.143 5.143-13.143c13.714-18.857 90.857-102.286 110.286-102.286 5.143 0 10.286 2.286 14.857 5.714l78.857 61.143c16.571-8.571 34.286-16 52-21.714 4-34.857 7.429-72 16.571-106.286 2.286-9.143 10.286-16 20.571-16h126.857c10.286 0 19.429 7.429 20.571 17.143l16 105.143c17.714 5.714 34.857 12.571 51.429 21.143l81.143-61.143c3.429-3.429 8.571-5.143 13.714-5.143s10.286 2.286 14.286 5.714c30.286 28 70.286 63.429 94.286 97.143 2.857 3.429 4 8 4 12.571 0 5.143-1.714 9.143-4.571 13.143-19.429 26.286-40.571 51.429-60 78.286 9.714 18.286 17.714 37.143 23.429 56l104.571 16c9.714 1.714 16.571 10.857 16.571 20.571z"
                  ></path>
                </svg>
                <span class="home-text07">Einstellungen</span>
              </div>
            </div>
          </div>
          <div class="home-container17"></div>
        </div>
      </div>
      <div class="home-container18">
        <div class="home-container19">
          <div class="home-container20">
            <div class="home-container21">
              <div class="home-container22">
                <div class="home-container23"><span>aktives Modell</span></div>
                <div class="home-container24">
                  <svg viewBox="0 0 1024 1024" class="home-icon26">
                    <path
                      d="M512 682.667h-341.333c-11.776 0-22.4-4.736-30.165-12.501s-12.501-18.389-12.501-30.165v-426.667c0-11.776 4.736-22.4 12.501-30.165s18.389-12.501 30.165-12.501h682.667c11.776 0 22.4 4.736 30.165 12.501s12.501 18.389 12.501 30.165v426.667c0 11.776-4.736 22.4-12.501 30.165s-18.389 12.501-30.165 12.501zM469.333 768v85.333h-128c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667h341.333c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-128v-85.333h298.667c35.328 0 67.413-14.379 90.496-37.504s37.504-55.168 37.504-90.496v-426.667c0-35.328-14.379-67.413-37.504-90.496s-55.168-37.504-90.496-37.504h-682.667c-35.328 0-67.413 14.379-90.496 37.504s-37.504 55.168-37.504 90.496v426.667c0 35.328 14.379 67.413 37.504 90.496s55.168 37.504 90.496 37.504z"
                    ></path>
                  </svg>
                  <select>
                    <option value="Option 1" selected>CNN</option>
                    <option value="Option 2">ANN</option>
                    <option value="Option 1">Option 1</option>
                    <option value="Option 2">Option 2</option>
                    <option value="Option 3">Option 3</option>
                  </select>
                </div>
              </div>
              <div class="home-container25">
                <div class="home-container26">
                  <div class="home-container27">
                    <span>Modell Guthaben</span>
                  </div>
                  <div class="home-container28">
                    <svg viewBox="0 0 1024 1024" class="home-icon28">
                      <path
                        d="M928 128h-832c-52.8 0-96 43.2-96 96v576c0 52.8 43.2 96 96 96h832c52.8 0 96-43.2 96-96v-576c0-52.8-43.2-96-96-96zM96 192h832c17.346 0 32 14.654 32 32v96h-896v-96c0-17.346 14.654-32 32-32zM928 832h-832c-17.346 0-32-14.654-32-32v-288h896v288c0 17.346-14.654 32-32 32zM128 640h64v128h-64zM256 640h64v128h-64zM384 640h64v128h-64z"
                      ></path>
                    </svg>
                    <input
                      type="text"
                      placeholder="Wert in Euro"
                      class="home-textinput input"
                    />
                  </div>
                </div>
              </div>
              <div class="home-container29">
                <div class="home-container30">
                  <div class="home-container31"><span>Traiding Tags</span></div>
                  <div class="home-container32">
                    <svg viewBox="0 0 1024 1024" class="home-icon30">
                      <path
                        d="M914.005 409.344l-275.84-278.827c-56.704-56.747-131.712-87.851-211.499-87.851s-154.795 31.104-211.2 87.509c-60.416 60.331-90.752 143.189-86.485 226.731-28.245 46.293-43.648 99.413-43.648 155.051 0 79.787 31.104 154.795 87.509 211.2l130.261 127.317 148.565 148.48c16.683 16.725 38.485 25.045 60.331 25.045s43.648-8.32 60.331-25.003l298.667-298.667c33.195-33.195 33.365-86.955 0.341-120.32l-3.968-4.011 46.293-46.336c33.195-33.195 33.365-86.955 0.341-120.32zM512 938.667l-149.248-149.205-129.579-126.635c-83.328-83.328-83.328-218.411 0-301.653 41.643-41.685 96.256-62.507 150.827-62.507s109.184 20.821 150.827 62.507l275.84 278.827-298.667 298.667zM595.499 301.184c-56.704-56.747-131.712-87.851-211.499-87.851-54.443 0-106.539 14.805-152.107 41.899 10.283-23.467 24.704-45.525 43.947-64.768 41.643-41.643 96.256-62.464 150.827-62.464s109.184 20.821 150.827 62.507l275.84 278.827-45.995 45.995-211.84-214.144z"
                      ></path>
                      <path
                        d="M384 447.957c35.413 0 64 28.672 64 64.043 0 35.413-28.587 63.957-64 63.957s-64-28.544-64-63.957c0-35.371 28.587-64.043 64-64.043M384 405.291c-58.795 0-106.667 47.872-106.667 106.709 0 58.795 47.872 106.624 106.667 106.624s106.667-47.829 106.667-106.624c0-58.837-47.872-106.709-106.667-106.709z"
                      ></path>
                    </svg>
                    <select>
                      <option value="Option 1" selected>
                        Umweltfreundlich
                      </option>
                      <option value="Option 2">Risiko</option>
                    </select>
                  </div>
                </div>
              </div>
              <div class="home-container33">
                <div class="home-container34">
                  <span>Los geht&apos;s!</span>
                </div>
                <div class="home-container35">
                  <svg viewBox="0 0 1024 1024" class="home-icon33">
                    <path
                      d="M480 64c-265.096 0-480 214.904-480 480s214.904 480 480 480c265.098 0 480-214.902 480-480s-214.902-480-480-480zM480 928c-212.078 0-384-171.922-384-384s171.922-384 384-384c212.076 0 384 171.922 384 384s-171.924 384-384 384z"
                    ></path>
                    <path
                      d="M670.824 644.34c-15.27-8.884-34.862-3.708-43.75 11.57-17.256 29.662-49.088 48.090-83.074 48.090h-128c-41.716 0-77.286-26.754-90.496-64h154.496c17.672 0 32-14.326 32-32s-14.328-32-32-32h-160v-64h160c17.672 0 32-14.328 32-32s-14.328-32-32-32h-154.496c13.21-37.246 48.78-64 90.496-64h128c33.986 0 65.818 18.426 83.074 48.090 8.888 15.276 28.478 20.456 43.752 11.568 15.276-8.888 20.456-28.476 11.568-43.752-28.672-49.288-81.702-79.906-138.394-79.906h-128c-77.268 0-141.914 55.056-156.78 128h-35.22c-17.672 0-32 14.328-32 32s14.328 32 32 32h32v64h-32c-17.672 0-32 14.326-32 32s14.328 32 32 32h35.22c14.866 72.944 79.512 128 156.78 128h128c56.692 0 109.72-30.62 138.394-79.91 8.888-15.276 3.708-34.864-11.57-43.75z"
                    ></path>
                  </svg>
                  <button type="button" class="button">Starte Traiding</button>
                </div>
              </div>
            </div>
          </div>
          <div class="home-container36">
            <div class="home-container37">
              <div class="home-container38">
                <div class="home-container39">
                  <DangerousHTML
                    html="<!DOCTYPE html>
  <html>
  <head>
    <title>Custom Graph</title>
    <style>
      canvas {
        width: 100%;
        height: 300px;
        border: 1px solid #ccc;
      }
    </style>
  </head>
  <body>
    <canvas id='chart' style='width:100%;max-width:700px'></canvas>
    <script>
      const xValues = ['Data Point 1', 'Data Point 2', 'Data Point 3', 'Data Point 4', 'Data Point 5'];
      const yValues = [10, 20, 30, 40, 50];
      const lineColor = 'red';
  
      const canvas = document.getElementById('chart');
      const ctx = canvas.getContext('2d');
  
      const maxYValue = Math.max(...yValues);
      const yScale = canvas.height / maxYValue;
      const xScale = canvas.width / (xValues.length - 1);
  
      // Draw the line chart
      ctx.beginPath();
      ctx.moveTo(0, canvas.height - yValues[0] * yScale);
      for (let i = 1; i < yValues.length; i++) {
        ctx.lineTo(i * xScale, canvas.height - yValues[i] * yScale);
      }
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 2;
      ctx.stroke();
  
      // Draw the x-axis
      ctx.beginPath();
      ctx.moveTo(0, canvas.height);
      ctx.lineTo(canvas.width, canvas.height);
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 1;
      ctx.stroke();
  
      // Draw the y-axis
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, canvas.height);
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 1;
      ctx.stroke();
  
      // Add labels to the x-axis
      for (let i = 0; i < xValues.length; i++) {
        ctx.fillText(xValues[i], i * xScale, canvas.height + 20);
      }
  
      // Add labels to the y-axis
      for (let i = 0; i <= maxYValue; i += 10) {
        ctx.fillText(i, -20, canvas.height - i * yScale);
      }
  
      // Add label to the x-axis
      ctx.fillText('X Axis Label', canvas.width / 2, canvas.height);
  
      // Add label to the y-axis
      ctx.save();
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Y Axis Label', -canvas.height / 2, 10);
      ctx.restore();
    </script>
  </body>
  </html>"
                  ></DangerousHTML>
                </div>
              </div>
            </div>
            <div class="home-container40">
              <div class="home-container41">
                <span class="home-text12">Besitzer</span>
                <span class="home-text13">Gültig bis</span>
                <span class="home-text14">Kartennummer</span>
              </div>
            </div>
          </div>
          <div class="home-container42">
            <div class="home-container43">
              <div class="home-container44">
                <div class="home-container45">
                  <DangerousHTML
                    html="<!DOCTYPE html>
  <html>
  <head>
    <title>Custom Graph</title>
    <style>
      svg {
        width: 100%;
        height: 300px;
        border: 1px solid #ccc;
      }
    </style>
  </head>
  <body>
    <svg id='chart'></svg>
    <script>
      const data = [10, 20, 30, 40, 50];
      const svg = document.getElementById('chart');
      const barWidth = 40;
      const barSpacing = 10;
      let x = barSpacing;
  
      data.forEach((value) => {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', 300 - value);
        rect.setAttribute('width', barWidth);
        rect.setAttribute('height', value);
        rect.setAttribute('fill', 'steelblue');
        svg.appendChild(rect);
        x += barWidth + barSpacing;
      });
    </script>
  </body>
  </html>"
                  ></DangerousHTML>
                </div>
              </div>
            </div>
            <div class="home-container46"></div>
            <div class="home-container47"></div>
          </div>
        </div>
      </div>
    </div>
    <div class="home-container48"></div>
  </div>
</template>

<script>
import DangerousHTML from "dangerous-html/vue";

export default {
  name: "Home",
  components: {
    DangerousHTML,
  },
  methods: {
    logout() {
      localStorage.removeItem("logegd_user_id");
      localStorage.removeItem("logged_user");
      this.$store.commit("setUser", null);
      this.$store.commit("setUserId", null);
      this.$router.push("/login");
    },
  },
  metaInfo: {
    title: "frontend",
    meta: [
      {
        property: "og:title",
        content: "frontend",
      },
    ],
  },
};
</script>

<style scoped>
.home-container {
  width: 100%;
  display: flex;
  overflow: auto;
  min-height: 100vh;
  align-items: center;
  flex-direction: column;
}
.home-header {
  width: 100%;
  display: flex;
  align-items: center;
  padding-top: var(--dl-space-space-oneandhalfunits);
  flex-direction: row;
  padding-bottom: var(--dl-space-space-oneandhalfunits);
  justify-content: flex-end;
  background-color: #ffffff;
}
.home-image {
  width: 340px;
  object-fit: cover;
}
.home-navbar-interactive {
  width: 90%;
  display: flex;
  position: sticky;
  max-width: var(--dl-size-size-maxwidth);
  align-items: center;
  padding-top: var(--dl-space-space-oneandhalfunits);
  padding-left: var(--dl-space-space-threeunits);
  border-radius: 50px;
  padding-right: var(--dl-space-space-threeunits);
  padding-bottom: var(--dl-space-space-oneandhalfunits);
  justify-content: space-between;
  background-color: var(--dl-color-gray-white);
}
.home-container01 {
  flex: 0 0 auto;
  width: 994px;
  height: 100px;
  display: flex;
  box-shadow: 5px 5px 10px 5px #000000;
  align-items: flex-end;
  flex-direction: column;
  justify-content: center;
}
.home-logo {
  font-size: 2em;
  align-self: flex-start;
  font-family: "Lato";
  font-weight: bold;
  margin-left: var(--dl-space-space-unit);
  text-transform: uppercase;
}
.home-icon {
  width: 35px;
  height: 101px;
  align-self: flex-end;
  margin-right: var(--dl-space-space-halfunit);
}
.home-container02 {
  flex: 0 0 auto;
  width: auto;
  height: auto;
  display: flex;
  align-items: space-between;
  flex-direction: column;
}
.home-profile {
  width: 144px;
  height: 82px;
  display: flex;
  align-items: center;
  padding-top: 5px;
  border-color: #d9d9d9;
  border-width: 1px;
  flex-direction: row;
  justify-content: flex-start;
  border-left-width: 0px;
  border-right-width: 0px;
  border-bottom-width: 0px;
}
.home-container03 {
  width: var(--dl-size-size-small);
  height: var(--dl-size-size-small);
  display: flex;
  align-items: center;
  border-radius: var(--dl-radius-radius-round);
  justify-content: center;
}
.home-image1 {
  width: 200px;
  object-fit: cover;
  margin-left: var(--dl-space-space-halfunit);
  border-radius: var(--dl-radius-radius-round);
}
.home-container04 {
  display: flex;
  align-items: flex-start;
  margin-left: var(--dl-space-space-halfunit);
  flex-direction: column;
  justify-content: space-between;
}
.home-text {
  font-size: 1.15rem;
  font-style: normal;
  font-weight: 600;
  margin-bottom: 0.25rem;
}
.home-text01 {
  color: var(--dl-color-gray-500);
  font-size: 0.75rem;
}
.home-desktop-menu {
  flex: 1;
  display: flex;
  justify-content: space-between;
}
.home-burger-menu {
  display: none;
}
.home-icon02 {
  width: var(--dl-size-size-xsmall);
  cursor: pointer;
  height: var(--dl-size-size-xsmall);
}
.home-mobile-menu {
  top: 0px;
  left: 0px;
  width: 100%;
  height: 100vh;
  display: none;
  padding: 32px;
  z-index: 100;
  position: absolute;
  border-radius: 50px;
  flex-direction: column;
  justify-content: space-between;
  background-color: var(--dl-color-gray-white);
}
.home-nav {
  display: flex;
  align-items: flex-start;
  flex-direction: column;
}
.home-top {
  width: 100%;
  display: flex;
  align-items: center;
  margin-bottom: var(--dl-space-space-threeunits);
  justify-content: space-between;
}
.home-logo1 {
  font-size: 2em;
  font-weight: bold;
}
.home-close-menu {
  display: flex;
  align-items: center;
  justify-content: center;
}
.home-icon04 {
  fill: var(--dl-color-gray-900);
  width: var(--dl-size-size-xsmall);
  cursor: pointer;
  height: var(--dl-size-size-xsmall);
}
.home-links {
  flex: 0 0 auto;
  display: flex;
  align-self: flex-start;
  align-items: flex-start;
  flex-direction: column;
}
.home-nav1 {
  color: var(--dl-color-gray-black);
  margin-bottom: var(--dl-space-space-unit);
}
.home-nav2 {
  color: var(--dl-color-gray-black);
  margin-bottom: var(--dl-space-space-unit);
}
.home-nav3 {
  color: var(--dl-color-gray-black);
  margin-bottom: var(--dl-space-space-unit);
}
.home-nav4 {
  color: var(--dl-color-gray-black);
  margin-bottom: var(--dl-space-space-unit);
}
.home-nav5 {
  color: var(--dl-color-gray-black);
  margin-bottom: var(--dl-space-space-unit);
}
.home-buttons {
  display: flex;
  margin-top: var(--dl-space-space-unit);
  align-items: center;
  flex-direction: row;
  justify-content: space-between;
}
.home-login {
  border-color: var(--dl-color-primary1-blue100);
  border-width: 1px;
  margin-right: var(--dl-space-space-twounits);
  padding-left: 1.5rem;
  border-radius: 30px;
  padding-right: 1.5rem;
}
.home-register {
  color: var(--dl-color-gray-white);
  transition: 0.3s;
  border-color: var(--dl-color-primary1-blue100);
  border-width: 1px;
  padding-left: 1.5rem;
  border-radius: 30px;
  padding-right: 1.5rem;
  background-color: var(--dl-color-primary1-blue100);
}
.home-register:hover {
  border-color: #849492ff;
  background-color: #849492ff;
}
.home-icon06 {
  width: var(--dl-size-size-xsmall);
  height: var(--dl-size-size-xsmall);
  margin-right: var(--dl-space-space-twounits);
}
.home-icon08 {
  width: var(--dl-size-size-xsmall);
  height: var(--dl-size-size-xsmall);
  margin-right: var(--dl-space-space-twounits);
}
.home-icon10 {
  width: var(--dl-size-size-xsmall);
  height: var(--dl-size-size-xsmall);
}
.home-container05 {
  flex: 0 0 auto;
  width: auto;
  height: 80vh;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  background-color: #b9e3e6;
}
.home-container06 {
  flex: 0 0 auto;
  width: auto;
  height: 80vh;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
}
.home-sidebar {
  width: 316px;
  height: 923px;
  display: flex;
  padding: var(--dl-space-space-twounits);
  box-shadow: 5px 5px 10px 0px #d4d4d4;
  align-items: center;
  flex-direction: column;
  justify-content: flex-start;
  background-color: #ffffff;
}
.home-container07 {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  flex-direction: row;
  justify-content: flex-end;
}
.home-container08 {
  width: 152px;
  height: 475px;
  display: flex;
  align-self: flex-start;
  align-items: flex-start;
  flex-direction: column;
  justify-content: flex-start;
}
.home-container09 {
  flex: 0 0 auto;
  width: auto;
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}
.home-icon12 {
  width: var(--dl-size-size-xsmall);
  height: var(--dl-size-size-xsmall);
  margin-right: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-text02 {
  color: rgb(80, 245, 100);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-container10 {
  flex: 0 0 auto;
  width: auto;
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}
.home-icon14 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-text03 {
  color: #50f564;
  margin-bottom: var(--dl-space-space-twounits);
}
.home-container11 {
  flex: 0 0 auto;
  width: auto;
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}
.home-icon16 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-text04 {
  color: rgb(80, 245, 100);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-container12 {
  flex: 0 0 auto;
  width: auto;
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}
.home-icon18 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-navlink {
  color: rgb(80, 245, 100);
  margin-bottom: var(--dl-space-space-twounits);
  text-decoration: none;
}
.home-container13 {
  flex: 0 0 auto;
  width: auto;
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}
.home-icon20 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-text05 {
  color: rgb(80, 245, 100);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-container14 {
  flex: 0 0 auto;
  width: auto;
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}
.home-container15 {
  flex: 0 0 auto;
  width: XSmall;
  height: auto;
  display: flex;
  align-items: center;
  flex-direction: column;
}
.home-icon22 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-text06 {
  color: rgb(80, 245, 100);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-container16 {
  flex: 0 0 auto;
  width: auto;
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}
.home-icon24 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-text07 {
  color: rgb(80, 245, 100);
  margin-bottom: var(--dl-space-space-twounits);
}
.home-container17 {
  flex: 0 0 auto;
  width: auto;
  border: 2px dashed rgba(120, 120, 120, 0.4);
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
}
.home-container18 {
  flex: 0 0 auto;
  width: 1244px;
  height: 100%;
  display: flex;
  align-items: center;
  border-color: var(--dl-color-gray-black);
  border-width: 1px;
  flex-direction: row;
  justify-content: flex-start;
  background-color: #b9e3e6;
}
.home-container19 {
  width: 0px;
  height: auto;
  display: flex;
  align-items: flex-start;
  flex-direction: column;
}
.home-container20 {
  flex: 0 0 auto;
  width: auto;
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
}
.home-container21 {
  flex: 0 0 auto;
  width: 1247px;
  display: flex;
  align-items: center;
  flex-direction: row;
  justify-content: space-between;
}
.home-container22 {
  flex: 0 0 auto;
  width: 200px;
  height: 87px;
  display: flex;
  box-shadow: 5px 5px 10px 0px #000000;
  align-items: center;
  margin-left: var(--dl-space-space-unit);
  border-color: var(--dl-color-gray-black);
  border-style: groove;
  border-width: 1px;
  flex-direction: column;
  justify-content: center;
  background-color: #d9d9d9;
}
.home-container23 {
  width: 200px;
  height: 27px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
}
.home-container24 {
  flex: 0 0 auto;
  width: 100%;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.home-icon26 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
}
.home-container25 {
  width: 214px;
  height: 89px;
  display: flex;
  align-items: center;
  flex-direction: row;
  justify-content: center;
  background-color: #d9d9d9;
}
.home-container26 {
  width: 100%;
  height: 87px;
  display: flex;
  box-shadow: 5px 5px 10px 0px #000000;
  align-items: center;
  border-color: var(--dl-color-gray-black);
  border-style: groove;
  border-width: 1px;
  flex-direction: column;
  justify-content: center;
  background-color: #d9d9d9;
}
.home-container27 {
  width: 200px;
  height: 27px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
}
.home-container28 {
  flex: 0 0 auto;
  width: 266px;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.home-icon28 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
}
.home-textinput {
  width: 174px;
  height: 24px;
}
.home-container29 {
  width: 200px;
  height: 89px;
  display: flex;
  align-items: center;
  flex-direction: row;
  justify-content: center;
  background-color: #d9d9d9;
}
.home-container30 {
  flex: 0 0 auto;
  width: 200px;
  height: 87px;
  display: flex;
  box-shadow: 5px 5px 10px 0px #000000;
  align-items: center;
  border-color: var(--dl-color-gray-black);
  border-style: groove;
  border-width: 1px;
  flex-direction: column;
  justify-content: center;
  background-color: #d9d9d9;
}
.home-container31 {
  width: 200px;
  height: 27px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
}
.home-container32 {
  flex: 0 0 auto;
  width: 266px;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.home-icon30 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
}
.home-container33 {
  flex: 0 0 auto;
  width: 200px;
  height: 87px;
  display: flex;
  box-shadow: 5px 5px 10px 0px #000000;
  align-items: center;
  border-color: var(--dl-color-gray-black);
  border-style: groove;
  border-width: 1px;
  margin-right: var(--dl-space-space-unit);
  flex-direction: column;
  justify-content: center;
  background-color: #d9d9d9;
}
.home-container34 {
  width: 200px;
  height: 27px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
}
.home-container35 {
  flex: 0 0 auto;
  width: 266px;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.home-icon33 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
}
.home-container36 {
  width: 1249px;
  height: 323px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
}
.home-container37 {
  flex: 0 0 auto;
  width: 717px;
  height: 298px;
  display: flex;
  align-self: center;
  margin-top: var(--dl-space-space-unit);
  align-items: center;
  margin-left: var(--dl-space-space-unit);
  margin-right: var(--dl-space-space-unit);
  margin-bottom: var(--dl-space-space-unit);
  flex-direction: column;
  justify-content: center;
  background-color: #eff0f2;
}
.home-container38 {
  width: 721px;
  height: 301px;
  box-shadow: 5px 5px 10px 0px #000000;
  border-color: var(--dl-color-gray-black);
  border-style: groove;
  border-width: 1px;
}
.home-container39 {
  display: contents;
}
.home-container40 {
  flex: 0 0 auto;
  width: 408px;
  height: 235px;
  display: flex;
  align-self: center;
  align-items: space-between;
  flex-direction: column;
}
.home-container41 {
  width: 389px;
  height: 100%;
  display: flex;
  align-self: center;
  box-shadow: 5px 5px 10px 0px #000000;
  align-items: center;
  margin-right: var(--dl-space-space-unit);
  flex-direction: column;
  background-size: contain;
  justify-content: flex-start;
  background-image: url("/oie_q3itd7eu4hf6-300h.jpg");
}
.home-text12 {
  color: #ffffff;
  font-size: 25px;
  align-self: flex-start;
  margin-top: var(--dl-space-space-halfunit);
  margin-left: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-threeunits);
}
.home-text13 {
  color: #ffffff;
  font-size: 25px;
  align-self: flex-start;
  font-style: normal;
  font-weight: 400;
  margin-left: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-oneandhalfunits);
}
.home-text14 {
  color: #ffffff;
  font-size: 25px;
  align-self: flex-start;
  margin-left: var(--dl-space-space-halfunit);
  margin-bottom: 56px;
}
.home-container42 {
  width: 1249px;
  height: 325px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.home-container43 {
  flex: 0 0 auto;
  width: 296px;
  height: 298px;
  display: flex;
  align-items: center;
  margin-left: var(--dl-space-space-unit);
  border-color: var(--dl-color-gray-black);
  border-style: groove;
  border-width: 1px;
  flex-direction: column;
  justify-content: center;
  background-color: #eff0f2;
}
.home-container44 {
  width: 291px;
  height: 298px;
  box-shadow: 5px 5px 10px 0px #000000;
}
.home-container45 {
  display: contents;
}
.home-container46 {
  flex: 0 0 auto;
  width: 200px;
  border: 2px dashed rgba(120, 120, 120, 0.4);
  height: 100%;
  display: flex;
  box-shadow: 5px 5px 10px 0px #000000;
  align-items: flex-start;
  margin-right: var(--dl-space-space-unit);
  flex-direction: column;
  background-color: #eff0f2;
}
.home-container47 {
  flex: 0 0 auto;
  width: 200px;
  border: 2px dashed rgba(120, 120, 120, 0.4);
  height: 100%;
  display: flex;
  box-shadow: 5px 5px 10px 0px #000000;
  align-items: flex-start;
  margin-right: var(--dl-space-space-unit);
  flex-direction: column;
  background-color: #eff0f2;
}
.home-container48 {
  flex: 0 0 auto;
  width: auto;
  border: 2px dashed rgba(120, 120, 120, 0.4);
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}
@media (max-width: 991px) {
  .home-profile {
    padding-left: var(--dl-space-space-halfunit);
    padding-right: var(--dl-space-space-halfunit);
  }
  .home-container03 {
    margin-left: 0px;
  }
}
@media (max-width: 767px) {
  .home-navbar-interactive {
    padding-left: var(--dl-space-space-twounits);
    padding-right: var(--dl-space-space-twounits);
  }
  .home-desktop-menu {
    display: none;
  }
  .home-burger-menu {
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .home-mobile-menu {
    padding-top: var(--dl-space-space-oneandhalfunits);
    padding-left: var(--dl-space-space-twounits);
    padding-right: var(--dl-space-space-twounits);
    padding-bottom: var(--dl-space-space-oneandhalfunits);
  }
  .home-nav1 {
    margin-bottom: var(--dl-space-space-unit);
  }
  .home-nav2 {
    margin-left: 0;
    margin-bottom: var(--dl-space-space-unit);
  }
  .home-nav3 {
    margin-left: 0;
    margin-bottom: var(--dl-space-space-unit);
  }
  .home-nav4 {
    margin-left: 0;
    margin-bottom: var(--dl-space-space-unit);
  }
  .home-nav5 {
    margin-left: 0;
    margin-bottom: var(--dl-space-space-unit);
  }
  .home-icon12 {
    display: flex;
  }
}
@media (max-width: 479px) {
  .home-navbar-interactive {
    width: 90%;
    padding: var(--dl-space-space-unit);
  }
  .home-logo {
    font-size: 1.5em;
  }
  .home-mobile-menu {
    padding: 16px;
  }
  .home-logo1 {
    font-size: 1.5em;
  }
  .home-icon04 {
    fill: var(--dl-color-gray-900);
  }
}
</style>
