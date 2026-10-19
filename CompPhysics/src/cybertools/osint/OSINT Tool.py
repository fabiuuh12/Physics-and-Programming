import requests

def check_username(username):
    sites = {
        "GitHub": f"https://github.com/{username}",
        "Reddit": f"https://www.reddit.com/user/{username}",
        "Instagram": f"https://www.instagram.com/{username}/",
        "Twitter (X)": f"https://x.com/{username}",
        "Facebook": f"https://www.facebook.com/{username}",
        "TikTok": f"https://www.tiktok.com/@{username}",
        "Pinterest": f"https://www.pinterest.com/{username}/",
        "LinkedIn": f"https://www.linkedin.com/in/{username}",
        "Steam": f"https://steamcommunity.com/id/{username}",
        "Twitch": f"https://www.twitch.tv/{username}",
        "Medium": f"https://medium.com/@{username}",
        "YouTube": f"https://www.youtube.com/@{username}",
        "SoundCloud": f"https://soundcloud.com/{username}",
        "DeviantArt": f"https://www.deviantart.com/{username}",
        "About.me": f"https://about.me/{username}",
        "ProductHunt": f"https://www.producthunt.com/@{username}",
        "Replit": f"https://replit.com/@{username}",
        "Ko-fi": f"https://ko-fi.com/{username}",
        "WordPress": f"https://{username}.wordpress.com",
        "GitLab": f"https://gitlab.com/{username}",
        "Fiverr": f"https://www.fiverr.com/{username}",
        "Tumblr": f"https://{username}.tumblr.com",
        "Vimeo": f"https://vimeo.com/{username}",
        "Dribbble": f"https://dribbble.com/{username}",
        "CashApp": f"https://cash.app/${username}"
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    print(f"\n🔍 Scanning for username: {username}\n")
    for site, url in sites.items():
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                print(f"[+] Found on {site}: {url}")
            elif response.status_code == 404:
                print(f"[-] Not on {site}")
            else:
                print(f"[?] {site}: Status {response.status_code}")
        except requests.RequestException as e:
            print(f"[!] Error with {site}: {e}")

if __name__ == "__main__":
    user = input("Enter a username to scan: ").strip()
    check_username(user)
