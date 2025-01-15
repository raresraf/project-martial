#include <iostream>

using namespace std;

int main() {
  int t;
  cin >> t;
  cin.ignore();
  while (t--) {
    string str;
    cin >> str;
    if (str.length() <= 10)
      cout << str << endl;
    else
      cout << str[0] << str.length() - 2 << str[str.length() - 1] << endl;
  }

  return 0;
}
