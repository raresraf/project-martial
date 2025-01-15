#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  string str;
  cin >> n;

  while (n-- > 0) {
    cin >> str;

    if (str.length() > 10) {
      cout << str.at(0) << str.length() - 2 << str.at(str.length() - 1) << endl;
    } else {
      cout << str << endl;
    }
  }
  return 0;
}
