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

  string s;
  cin >> s;
  int cnt1{};
  int cnt0{};
  for (auto i : s) {
    if (i == '1') {
      cnt1++;
      cnt0 = 0;
    }

    elif (i == '0') {
      cnt1 = 0;
      cnt0++;
    }

    if (cnt1 >= 7 || cnt0 >= 7) {
      cout << "YES" << endl;
      return 0;
    }
  }
  cout << "NO" << endl;
  return 0;
}
// TASKKILL /F /IM main.exe
