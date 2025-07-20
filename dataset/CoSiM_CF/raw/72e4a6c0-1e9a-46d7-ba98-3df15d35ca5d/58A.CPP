#include "bits/stdc++.h"
// PRAY :ORZ FENCING: orz orz orz orz orz
// PRAY :DOLPHINGARLIC: orz orz orz orz orz
// PRAY :DORI: orz orz orz orz orz orz orz
// PRAY :SAHIL KUCHLOUS: orz orz orz orz orz
// pray :stefan: orz orz orz orz orz orz orz
// pray :foshy: orz orz orz orz orz orz orz
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define elif else if

using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(0);

  int cnt{};
  const string h = "hello";
  string s;
  cin >> s;
  for (auto i : s) {
    if (i == h[cnt])
      cnt++;
  }
  cout << (cnt == 5 ? "YES\n" : "NO\n");
  return 0;
}
// TASKKILL /F /IM main.exe
/*
8:56 PM
9/30/2020 m/dd/yyyy
*/
