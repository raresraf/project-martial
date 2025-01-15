#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int ambil(vector<int> &coin, int total) {
  int take = 0;
  int jumlah = 0;
  while (total - jumlah >= jumlah) {
    jumlah = 0;
    for (int i = 0; i < take + 1; i++) {
      jumlah += coin[i];
    }
    take++;
  }
  return take;
}
bool desc(int i, int j) { return i > j; }

int main() {
  int n, a, total = 0;
  cin >> n;
  vector<int> coins;
  for (int i = 0; i < n; i++) {
    cin >> a;
    total += a;
    coins.push_back(a);
  }

  sort(coins.begin(), coins.end(), desc);
  cout << ambil(coins, total);
}
