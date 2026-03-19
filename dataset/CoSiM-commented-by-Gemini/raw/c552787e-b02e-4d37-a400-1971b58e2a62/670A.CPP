// Problem: Given a number of days n, find the minimum and maximum possible number of weekend days (Saturdays and Sundays).
// This is a competitive programming problem that involves calendar-based logic.
// The approach is to calculate the number of weekend days in the full weeks and then handle the remaining days.
// Time Complexity: O(1)
// Space Complexity: O(1)
#include <bits/stdc++.h>

using namespace std;

int main() {
  long int n, minimum, maximum;
  cin >> n;

  // Calculate the base number of weekend days from the full weeks.
  // Each full week has 2 weekend days.
  minimum = ((n / 7) * 2);
  maximum = ((n / 7) * 2);

  // To maximize the number of weekend days, we assume the period starts on a Saturday.
  // If there's at least one remaining day, it can be a Saturday.
  if (n % 7 > 0)
    maximum++;
  // If there are at least two remaining days, they can be a Saturday and a Sunday.
  if (n % 7 > 1)
    maximum++;
  
  // To minimize the number of weekend days, we assume the period starts on a Monday.
  // The first 5 remaining days will be weekdays.
  // If there are 6 remaining days, one of them must be a Saturday.
  if (n % 7 == 6)
    ++minimum;

  cout << minimum << " " << maximum;
}
