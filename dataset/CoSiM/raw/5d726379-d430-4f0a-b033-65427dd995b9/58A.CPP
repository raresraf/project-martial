#include <iostream>

using namespace std;

int main() {
  int len, i, a;
  string str, w;
  a = 0;
  w = "hello";
  cin >> str;
  len = str.size();

  for (i = 0; i < len; i++) {
    if (str[i] == w[a]) {
      a++;
    }
  }
  if (a == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}
