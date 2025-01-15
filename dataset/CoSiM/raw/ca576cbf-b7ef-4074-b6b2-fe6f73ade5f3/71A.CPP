#include <iostream>
#include <sstream>
#include <string>
using namespace std;

int main() {
  int i, n, length;
  string str, num;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> str;
    length = str.size();
    if (length > 10) {
      stringstream numss;
      numss << length - 2;
      numss >> num;
      str.replace(1, length - 2, num);
    }
    cout << str << endl;
  }
  return 0;
}
