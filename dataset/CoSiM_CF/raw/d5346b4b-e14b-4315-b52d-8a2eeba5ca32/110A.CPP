#include <iostream>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  string s;
  cin >> s;

  int lucky_numbers = 0;

  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '4' || s[i] == '7') {
      lucky_numbers++;
    }
  }

  if (lucky_numbers == 4 || lucky_numbers == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
