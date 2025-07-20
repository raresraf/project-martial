// 29.01.19 @odnaks
#include <iostream>

using namespace std;

int main() {
  long long int n;
  int c = 0;
  int c1 = 0;
  int c2 = 0;
  cin >> n;

  for (long long int i = n; i > 0; i /= 10)
    if (i % 10 == 4 || i % 10 == 7)
      c++;
  for (int i = c; i > 0; i /= 10) {
    if (i % 10 == 4 || i % 10 == 7)
      c1++;
    c2++;
  }
  if (c1 == c2 && c1 > 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}