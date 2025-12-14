#include <bits/stdc++.h>

using namespace std;

int main() {
  long int n, minimum, maximum;
  cin >> n;
  minimum = ((n / 7) * 2);
  maximum = ((n / 7) * 2);
  if (n % 7 > 0)
    maximum++;
  if (n % 7 > 1)
    maximum++;
  if (n % 7 == 6)
    ++minimum;
  cout << minimum << " " << maximum;
}
