#include <iostream>
#include <string.h>
using namespace std;

int main() {
  char a[100], b[100];
  cin >> a >> b;

  int res = strcasecmp(a, b);

  if (res < 0)
    res /= -1 * res;
  else if (res > 0)
    res /= res;

  cout << res << endl;

  return 0;
}