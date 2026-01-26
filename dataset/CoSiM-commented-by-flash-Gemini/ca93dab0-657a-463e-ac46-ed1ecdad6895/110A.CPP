#include <iostream>
using namespace std;
void luck_check(string x);
int main() {
  string num;
  cin >> num;
  luck_check(num);
  void luck_check();
  return 0;
}
void luck_check(string x) {
  unsigned long c = 0;
  for (unsigned long i = 0; i < x.size(); i++) {
    if (x[i] == '7' || x[i] == '4') {
      c++;
    }
  }
  if (c == 4 || c == 7) {
    cout << "YES";
  } else
    cout << "NO";
}
