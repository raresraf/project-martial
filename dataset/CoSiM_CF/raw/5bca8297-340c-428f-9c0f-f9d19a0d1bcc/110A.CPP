#include <iostream>
using namespace std;
int main() {

  long long int n;
  int flag = 0;
  cin >> n;
  int count = 0;

  while (n > 0) {

    if ((n % 10 == 4) || (n % 10 == 7))
      count++;
    n = n / 10;
  }
  while (count > 0) {

    if ((count % 10 == 4) || (count % 10 == 7)) {
      count = count / 10;
      flag = 1;
    } else
      break;
  }
  if (count == 0 && flag)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
