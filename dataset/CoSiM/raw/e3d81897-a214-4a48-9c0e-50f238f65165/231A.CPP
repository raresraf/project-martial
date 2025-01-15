#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main()

{
  long long int n;
  cin >> n;
  long long int count = 0;
  for (int i = 0; i < n; i++) {
    int petya, vasya, tonya;
    cin >> petya >> vasya >> tonya;
    int sum = petya + vasya + tonya;
    if (sum >= 2)
      count++;
  }
  cout << count << endl;
}
