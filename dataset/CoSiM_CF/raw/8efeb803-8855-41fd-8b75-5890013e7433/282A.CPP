#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  int n, x = 0;
  cin >> n;

  string statement, lowSubstr, highSubstr;
  for (int i = 0; i != n; ++i) {
    cin >> statement;
    lowSubstr = statement.substr(0, 2);
    highSubstr = statement.substr(1, 3);
    if (lowSubstr == "++" || highSubstr == "++") {
      x++;
    } else if (lowSubstr == "--" || highSubstr == "--") {
      x--;
    } else {
      continue;
    }
  }
  cout << x;
  return 0;
}
