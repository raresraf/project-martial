#include <iostream>
using namespace std;
int main() {
  int i, num, result = 0;
  char ch, a;
  cin >> num;
  for (i = 1; i <= num; i++) {
    cin >> a >> ch >> a;
    if (ch == '+')
      result++;
    else
      result--;
  }
  cout << result;

  return 0;
}
