#include <iostream>
#include <string>
using namespace std;

int main() {
  int t, n, counter = 0;
  string str, result = "";
  cin >> t;
  for (int i = 0; i < t; i++) {
    cin >> str;
    n = str.length();
    if (n <= 10)
      result = str;
    else
      result = str[0] + to_string(n - 2) + str[n - 1];
    cout << result << endl;
  }
  return (0);
}
