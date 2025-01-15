#include <iostream>
#include <string>
using namespace std;
int main() {
  int one = 0, zero = 0, i;
  string position;
  getline(cin, position);
  for (i = 0; i < position.length(); i++) {
    if (position[i] == '0') {
      zero++;
      one = 0;
    } else {
      one++;
      zero = 0;
    }
    if (one >= 7 || zero >= 7)
      break;
  }
  if (one >= 7 || zero >= 7)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
