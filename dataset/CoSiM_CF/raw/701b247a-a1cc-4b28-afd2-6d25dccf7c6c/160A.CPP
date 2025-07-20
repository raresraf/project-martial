#include <algorithm>
#include <array>
#include <functional>
#include <iostream>

using namespace std;

void mySort(int a[], unsigned n) { std::sort(a, a + n); }

void descSort(int a[], unsigned n) { std::sort(a, a + n, greater<int>()); }

int main() {

  int n;
  cin >> n;
  int sum = 0;
  int me = 0;
  int cpt = 0;
  int t[n];

  for (int i = 0; i < n; i++) {
    cin >> t[i];
    sum += t[i];
  }
  descSort(t, n);
  int j = 0;

  while (me <= (sum / 2)) {
    me += t[j];
    j++;
    cpt++;
  }

  cout << cpt;

  return 0;
}