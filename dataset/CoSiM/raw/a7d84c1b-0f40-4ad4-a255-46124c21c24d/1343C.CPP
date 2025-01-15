#include <bits/stdc++.h>
#define endl "\n"

using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);

  int t;
  cin >> t;

  long int n;
  for (int i = 0; i < t; i++) {
    cin >> n;

    vector<long long int> a(n), b;

    for (int j = 0; j < n; j++) {
      cin >> a[j];
    }

    long long int maior = a[0];
    long long int soma = 0;
    for (int j = 1; j < n; j++) {
      if (a[j] < 0 && a[j - 1] < 0 || a[j] > 0 && a[j - 1] > 0) {
        if (a[j] > maior) {
          maior = a[j];
        }
      } else {
        b.push_back(maior);
        soma += maior;
        maior = a[j];
      }
    }
    b.push_back(maior);
    soma += maior;

    cout << soma << endl;
  }
}