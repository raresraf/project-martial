#include <iostream>
using namespace std;
int check(long long int a) {
  int z;
  int b[2] = {0};
  while (a != 0) {
    z = a % 10;
    a /= 10;
    if (z == 7)
      b[0] = 1;
    else if (z == 4)
      b[1] = 1;
    if (z != 4 && z != 7)
      return 1;
  }
  if (b[0] == 0 || b[1] == 0)
    return 1;

  return 0;
}
int check1(long long int a) {
  int z, count1 = 0;
  while (a != 0) {
    z = a % 10;
    a /= 10;
    if (z == 4 || z == 7)
      count1++;
  }
  if (count1 == 4 || count1 == 7)
    return 0;
  return 1;
}
int main() {
  long long int a;
  cin >> a;
  int b = check(a);
  if (b) {
    if (check1(a)) {
      cout << "NO";
    } else
      cout << "YES";
  } else {
    if (check1(a))
      cout << "NO";
    else
      cout << "YES";
  }
  return 0;
}