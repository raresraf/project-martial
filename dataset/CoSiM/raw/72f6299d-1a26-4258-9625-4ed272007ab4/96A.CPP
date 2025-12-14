#include <iostream>
#include <string>
using namespace std;

int main() {

  string s;
  cin >> s;
  int counter = 1;
  for (int i = 1; i < s.length(); i++) {

    if (s[i - 1] == s[i]) {
      counter++;
      if (counter == 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else {
      counter = 1;
    }
  }

  if (counter == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
