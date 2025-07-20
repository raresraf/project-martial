
#include <iostream>
#include <math.h>
#define ll long long
using namespace std;
int main() {
  ll int n;
  cin >> n;
  ll int a[13] = {0};
  for (int i = 1; i < 13; i++) {
    a[i] = pow(10, i) - 1;
  }
  for (int i = 2; i < 13; i++) {
    for (int j = 1; j <= i - 1; j++) {
      a[i] = a[i] - a[j];
    }
  }
  //    for(int i=1;i<13;i++)
  //    {
  //        cout<<a[i]<<endl;
  //    }
  // value in array is to set

  ll int count = 0;
  for (int i = 1; i < 13; i++) {
    if (n - a[i] >= 0) {
      count += a[i] * i;
      n = n - a[i];
    } else {
      count += n * i;
      break;
    }
  }
  cout << count << endl;
}
