// solutions.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see buil4d output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add
//   Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project
//   and select the .sln file
#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <queue>
#include <stack>
#include <vector>

#define mp make_pair
#define pb push_back
#define f first
#define s second

using namespace std;

typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> coord;
typedef vector<int> veci;
typedef vector<coord> coords;
const int mod = 1000000007;

int main() {
  ios_base::sync_with_stdio(false);
  int amt, a, b, ppl(0), max(0);
  cin >> amt;
  while (amt--) {
    cin >> a >> b;
    ppl += b - a;
    if (ppl > max) {
      max = ppl;
    }
  }
  cout << max;
}