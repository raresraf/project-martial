#include <iostream>
#include <string>
using namespace std;

int main() {

  string number;
  long long num = 0;
  cin >> number;
  for (char &c : number) {
    if (c == '7' || c == '4')
      num++;
  }
  number = to_string(num);
  bool faro = false;
  for (char &c : number) {
    if (c != '7' && c != '4') {
      faro = true;
      break;
    }
  }
  if (faro)
    cout << "NO";
  else
    cout << "YES";

  return 0;
}
