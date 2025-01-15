#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  string str;

  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> str;
    if (str.length() <= 10) {
      cout << str << endl;
    } else {
      cout << str[0] + to_string(str.length() - 2) + str[str.length() - 1]
           << endl;
    }
  }
}
