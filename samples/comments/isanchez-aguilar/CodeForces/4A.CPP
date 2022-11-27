/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 4A - Watermelon
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  int weight;
  cin >> weight;
  if (weight <= 2)
    cout << "NO\n";
  else if (weight % 2)
    cout << "NO\n";
  else
    cout << "YES\n";
  return 0;
}