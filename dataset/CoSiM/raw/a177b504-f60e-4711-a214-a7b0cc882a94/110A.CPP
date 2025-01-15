#include <iostream>
#include <string>
using namespace std;
int main() {
  string l;
  cin >> l;
  int count = 0;
  for (int i = 0; i < l.size(); i++) {
    if (l[i] == '4' || l[i] == '7') {
      count++;
    }
  }
  if (count == 4 || count == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}