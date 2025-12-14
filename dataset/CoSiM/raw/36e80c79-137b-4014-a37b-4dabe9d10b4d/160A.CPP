#include <algorithm>
#include <iostream>
using namespace std;

bool cmp(int a, int b) { return a > b; }

int main() {
  int n, c[101], sum = 0;
  cin >> n;

  for (int i = 0; i < n; i++)
    cin >> c[i];
  for (int i = 0; i < n; i++)
    sum += c[i];
  ;

  sort(c, c + n, cmp);

  if (sum % 2 == 1)
    sum += 1;
  sum /= 2;

  int taken = 0, inHand = 0;
  for (int i = 0; i < n; i++) {
    inHand += c[i];
    taken++;

    if (inHand > sum)
      break;
  }

  cout << taken << endl;
  return 0;
}
