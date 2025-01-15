#include <iostream>
//#include<algorithm>
using namespace std;
void descendsort(int ara[], int n);
int main() {
  int n, i, a[105], sum = 0, mymonney = 0;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  // sort(a+n,a);
  descendsort(a, n);
  for (i = 0; i < n; i++) {
    mymonney += a[i];
    if (mymonney > (sum / 2))
      break;
  }
  cout << i + 1 << endl;
  return 0;
}
void descendsort(int ara[], int n) {
  for (int i = 0; i < n - 1; i++) {
    for (int j = i; j < n; j++) {
      if (ara[i] < ara[j])
        swap(ara[i], ara[j]);
    }
  }
}
