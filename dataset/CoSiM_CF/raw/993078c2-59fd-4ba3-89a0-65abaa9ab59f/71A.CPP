#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

int main() {
  string a, d;
  vector<string> dope;
  stringstream str1;
  int b, n;

  cin >> b;
  for (int i = 0; i < b; i++) {
    cin >> a;
    n = a.length();

    if (n > 10) {
      str1 << n - 2;
      d = a[0] + str1.str() + a[n - 1];
      dope.push_back(d);
      str1.str("");

    } else {
      dope.push_back(a);
    }
  }
  for (int i = 0; i < b; i++) {
    cout << dope[i] << endl;
  }

  return 0;
}
