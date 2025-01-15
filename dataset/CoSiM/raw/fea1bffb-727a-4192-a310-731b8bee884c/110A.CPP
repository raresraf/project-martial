#include <iostream>
#include <string.h>

using namespace std;

int main() {

  unsigned long long x;
  int t = 0;
  string str = "";

  cin >> x;

  // Int to String conversion
  str.append(to_string(x));
  // cout<<sizeof(x);
  for (int i = 0; i < str.length(); ++i) {
    if (str[i] == '4' || str[i] == '7') {
      t += 1;
    }
  }

  if (t == 4 || t == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}
