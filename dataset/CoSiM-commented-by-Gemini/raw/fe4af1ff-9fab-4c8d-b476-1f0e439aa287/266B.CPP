#include <iostream>
#include <string>

using namespace std;

int main() {

  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  int i;
  string temp;

  while (t--) {
    for (i = 0; i < n; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        temp += "GB";
        i++;
        continue;
      }
      temp += s[i];
    }
    s = temp;
    temp = "";
  }

  cout << s << endl;
}
