#include <iostream>
#include <string>

using namespace std;

int main() {

  int t, ans = 0;
  string s;

  cin >> t;

  while (t--) {
    cin >> s;
    if (s == "++X") {
      ++ans;
    } else if (s == "X++") {
      ans++;
    } else if (s == "--X") {
      --ans;
    } else {
      ans--;
    }
  }
  cout << ans << endl;
  return 0;
}
